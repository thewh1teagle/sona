package wav

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// Header contains WAV format metadata.
type Header struct {
	AudioFormat   uint16
	Channels      uint16
	SampleRate    uint32
	BitsPerSample uint16
}

// IsNative returns true if the WAV is already 16kHz mono 16-bit PCM
// and can be decoded without ffmpeg.
func (h Header) IsNative() bool {
	return h.AudioFormat == 1 && h.Channels == 1 && h.SampleRate == 16000 && h.BitsPerSample == 16
}

// ReadHeader reads the WAV header and seeks back to the start.
// Returns an error if the file is not a valid WAV.
func ReadHeader(r io.ReadSeeker) (Header, error) {
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return Header{}, fmt.Errorf("failed to read WAV header: %w", err)
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		r.Seek(0, io.SeekStart)
		return Header{}, fmt.Errorf("not a valid WAV file")
	}

	var h Header
	for {
		var chunkID [4]byte
		var chunkSize uint32
		if err := binary.Read(r, binary.LittleEndian, &chunkID); err != nil {
			r.Seek(0, io.SeekStart)
			return Header{}, fmt.Errorf("unexpected end of file: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &chunkSize); err != nil {
			r.Seek(0, io.SeekStart)
			return Header{}, fmt.Errorf("could not read chunk size: %w", err)
		}

		if string(chunkID[:]) == "fmt " {
			var fmtBuf [16]byte
			if _, err := io.ReadFull(r, fmtBuf[:]); err != nil {
				r.Seek(0, io.SeekStart)
				return Header{}, fmt.Errorf("failed to read fmt chunk: %w", err)
			}
			h.AudioFormat = binary.LittleEndian.Uint16(fmtBuf[0:2])
			h.Channels = binary.LittleEndian.Uint16(fmtBuf[2:4])
			h.SampleRate = binary.LittleEndian.Uint32(fmtBuf[4:8])
			h.BitsPerSample = binary.LittleEndian.Uint16(fmtBuf[14:16])
			r.Seek(0, io.SeekStart)
			return h, nil
		}

		if _, err := r.Seek(int64(chunkSize), io.SeekCurrent); err != nil {
			r.Seek(0, io.SeekStart)
			return Header{}, err
		}
	}
}

// Read parses a 16-bit PCM WAV from an io.ReadSeeker and returns float32 samples in [-1, 1].
// Assumes 16kHz mono — no resampling is done.
func Read(r io.ReadSeeker) ([]float32, error) {
	var header [12]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, fmt.Errorf("failed to read WAV header: %w", err)
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a valid WAV file")
	}

	var audioFormat, channels, bitsPerSample uint16

	var dataSize uint32
	for {
		var chunkID [4]byte
		var chunkSize uint32
		if err := binary.Read(r, binary.LittleEndian, &chunkID); err != nil {
			return nil, fmt.Errorf("unexpected end of file: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &chunkSize); err != nil {
			return nil, fmt.Errorf("could not read chunk size: %w", err)
		}

		switch string(chunkID[:]) {
		case "fmt ":
			var fmtBuf [16]byte
			if _, err := io.ReadFull(r, fmtBuf[:]); err != nil {
				return nil, fmt.Errorf("failed to read fmt chunk: %w", err)
			}
			audioFormat = binary.LittleEndian.Uint16(fmtBuf[0:2])
			channels = binary.LittleEndian.Uint16(fmtBuf[2:4])
			bitsPerSample = binary.LittleEndian.Uint16(fmtBuf[14:16])
			if chunkSize > 16 {
				if _, err := r.Seek(int64(chunkSize-16), io.SeekCurrent); err != nil {
					return nil, err
				}
			}
		case "data":
			dataSize = chunkSize
			goto readData
		default:
			if _, err := r.Seek(int64(chunkSize), io.SeekCurrent); err != nil {
				return nil, err
			}
		}
	}

readData:
	if audioFormat != 1 {
		return nil, fmt.Errorf("unsupported audio format %d (only PCM=1)", audioFormat)
	}
	if bitsPerSample != 16 {
		return nil, fmt.Errorf("unsupported bits per sample %d (only 16)", bitsPerSample)
	}

	nSamples := int(dataSize) / int(channels) / 2
	if nSamples == 0 {
		return nil, fmt.Errorf("audio file contains no samples")
	}
	raw := make([]int16, int(dataSize)/2)
	if err := binary.Read(r, binary.LittleEndian, raw); err != nil {
		return nil, fmt.Errorf("failed to read PCM data: %w", err)
	}

	samples := make([]float32, nSamples)
	for i := 0; i < nSamples; i++ {
		var sum float64
		for ch := 0; ch < int(channels); ch++ {
			sum += float64(raw[i*int(channels)+ch])
		}
		samples[i] = float32(sum / float64(channels) / math.MaxInt16)
	}
	return samples, nil
}

// ReadFile opens a WAV file by path and returns float32 samples.
func ReadFile(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}
