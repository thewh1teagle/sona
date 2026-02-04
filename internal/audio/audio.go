package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/thewh1teagle/sonara/internal/wav"
)

// findFFmpeg prefers a system ffmpeg from $PATH, then falls back
// to a bundled ffmpeg next to the current binary.
func findFFmpeg() (string, error) {
	path, err := exec.LookPath("ffmpeg")
	if err == nil {
		return path, nil
	}

	if exe, exErr := os.Executable(); exErr == nil {
		candidates := []string{
			filepath.Join(filepath.Dir(exe), "ffmpeg"),
			filepath.Join(filepath.Dir(exe), "ffmpeg.exe"),
		}
		for _, candidate := range candidates {
			if _, statErr := os.Stat(candidate); statErr == nil {
				return candidate, nil
			}
		}
	}

	return "", fmt.Errorf("ffmpeg not found: %w", err)
}

// convertWithFFmpeg writes the input to a temp file, runs ffmpeg to convert
// it to 16kHz mono s16le PCM via pipe output, and returns float32 samples.
func convertWithFFmpeg(r io.Reader, ffmpegPath string) ([]float32, error) {
	tmp, err := os.CreateTemp("", "sonara-*.audio")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmp.Name())
	defer tmp.Close()

	if _, err := io.Copy(tmp, r); err != nil {
		return nil, fmt.Errorf("failed to write temp file: %w", err)
	}
	tmp.Close()

	cmd := exec.Command(ffmpegPath,
		"-i", tmp.Name(),
		"-ar", "16000",
		"-ac", "1",
		"-f", "s16le",
		"-acodec", "pcm_s16le",
		"pipe:1",
	)
	cmd.Stderr = os.Stderr

	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffmpeg conversion failed: %w", err)
	}

	nSamples := len(out) / 2
	samples := make([]float32, nSamples)
	for i := 0; i < nSamples; i++ {
		sample := int16(binary.LittleEndian.Uint16(out[i*2 : i*2+2]))
		samples[i] = float32(float64(sample) / math.MaxInt16)
	}
	return samples, nil
}

// Read decodes audio from an io.ReadSeeker into float32 samples at 16kHz mono.
// If the input is a native 16kHz/mono/16-bit PCM WAV, it is decoded directly.
// Otherwise, ffmpeg is used to convert the audio.
func Read(r io.ReadSeeker) ([]float32, error) {
	h, err := wav.ReadHeader(r)
	if err == nil && h.IsNative() {
		return wav.Read(r)
	}

	// Not a native WAV — need ffmpeg
	r.Seek(0, io.SeekStart)
	ffmpegPath, err := findFFmpeg()
	if err != nil {
		if h.SampleRate != 0 {
			// It's a WAV but not native format
			return nil, fmt.Errorf("audio requires conversion (got %dHz %dch %dbit) but %w",
				h.SampleRate, h.Channels, h.BitsPerSample, err)
		}
		return nil, fmt.Errorf("unsupported audio format and %w", err)
	}

	return convertWithFFmpeg(r, ffmpegPath)
}

// ReadFile opens an audio file by path and returns float32 samples at 16kHz mono.
func ReadFile(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}
