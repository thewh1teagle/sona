package audio

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/thewh1teagle/sona/internal/wav"
)

var verbose bool

type ReadOptions struct {
	EnhanceAudio bool
}

func SetVerbose(v bool) {
	verbose = v
}

// findFFmpeg checks for ffmpeg in this order:
// 1. System ffmpeg from $PATH
// 2. SONA_FFMPEG_PATH env var (warns and continues if set but not found)
// 3. Bundled ffmpeg next to the current binary
func findFFmpeg() (string, error) {
	path, err := exec.LookPath("ffmpeg")
	if err == nil {
		return path, nil
	}

	if envPath := os.Getenv("SONA_FFMPEG_PATH"); envPath != "" {
		if _, statErr := os.Stat(envPath); statErr == nil {
			return envPath, nil
		}
		fmt.Fprintf(os.Stderr, "warning: SONA_FFMPEG_PATH set to %q but not found, continuing search\n", envPath)
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

// ConvertToNativeWav converts any audio file to a 16kHz mono 16-bit PCM WAV file
// on disk using ffmpeg. When enhanceAudio is true, a silence removal filter is applied.
func ConvertToNativeWav(inputPath, outputPath string, enhanceAudio bool) error {
	ffmpegPath, err := findFFmpeg()
	if err != nil {
		return err
	}

	args := []string{
		"-i", inputPath,
		"-ar", "16000",
		"-ac", "1",
	}
	if enhanceAudio {
		args = append(args, "-af", "silenceremove=stop_periods=-1:stop_duration=0.7:stop_threshold=-45dB")
	}
	args = append(args,
		"-acodec", "pcm_s16le",
		"-y",
		outputPath,
	)

	cmd := exec.Command(ffmpegPath, args...)
	if verbose {
		cmd.Stderr = os.Stderr
	} else {
		cmd.Stderr = io.Discard
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg WAV conversion failed: %w", err)
	}
	return nil
}

// Read decodes audio from an io.ReadSeeker into float32 samples at 16kHz mono.
// If the input is a native 16kHz/mono/16-bit PCM WAV, it is decoded directly.
// Otherwise, ffmpeg is used to convert the audio.
func Read(r io.ReadSeeker) ([]float32, error) {
	return ReadWithOptions(r, ReadOptions{})
}

func ReadWithOptions(r io.ReadSeeker, opts ReadOptions) ([]float32, error) {
	h, err := wav.ReadHeader(r)
	if err == nil && h.IsNative() && !opts.EnhanceAudio {
		return wav.Read(r)
	}

	// Not a native WAV (or enhancement requested) — need ffmpeg
	r.Seek(0, io.SeekStart)

	// Save to temp file for ffmpeg input
	tmp, err := os.CreateTemp("", "sona-*.audio")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmp.Name())

	if _, err := io.Copy(tmp, r); err != nil {
		tmp.Close()
		return nil, fmt.Errorf("failed to write temp file: %w", err)
	}
	tmp.Close()

	// Convert to native WAV via ffmpeg
	nativeWav := tmp.Name() + ".wav"
	if err := ConvertToNativeWav(tmp.Name(), nativeWav, opts.EnhanceAudio); err != nil {
		return nil, err
	}
	defer os.Remove(nativeWav)

	f, err := os.Open(nativeWav)
	if err != nil {
		return nil, fmt.Errorf("failed to open converted file: %w", err)
	}
	defer f.Close()
	return wav.Read(f)
}

// ReadFile opens an audio file by path and returns float32 samples at 16kHz mono.
func ReadFile(path string) ([]float32, error) {
	return ReadFileWithOptions(path, ReadOptions{})
}

func ReadFileWithOptions(path string, opts ReadOptions) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadWithOptions(f, opts)
}
