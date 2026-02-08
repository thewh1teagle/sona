package diarize

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// Segment represents a speaker segment from diarization.
type Segment struct {
	Start     float64 `json:"start"`
	End       float64 `json:"end"`
	SpeakerID int     `json:"speaker_id"`
}

// findDiarizer checks for sona-diarize in this order:
// 1. System sona-diarize from $PATH
// 2. SONA_DIARIZE_PATH env var (warns and continues if set but not found)
// 3. Bundled sona-diarize next to the current binary
func findDiarizer() (string, error) {
	path, err := exec.LookPath("sona-diarize")
	if err == nil {
		return path, nil
	}

	if envPath := os.Getenv("SONA_DIARIZE_PATH"); envPath != "" {
		if _, statErr := os.Stat(envPath); statErr == nil {
			return envPath, nil
		}
		fmt.Fprintf(os.Stderr, "warning: SONA_DIARIZE_PATH set to %q but not found, continuing search\n", envPath)
	}

	if exe, exErr := os.Executable(); exErr == nil {
		candidates := []string{
			filepath.Join(filepath.Dir(exe), "sona-diarize"),
			filepath.Join(filepath.Dir(exe), "sona-diarize.exe"),
		}
		for _, candidate := range candidates {
			if _, statErr := os.Stat(candidate); statErr == nil {
				return candidate, nil
			}
		}
	}

	return "", fmt.Errorf("sona-diarize not found: %w", err)
}

// Available reports whether the sona-diarize binary can be found.
func Available() bool {
	_, err := findDiarizer()
	return err == nil
}

// Diarize runs sona-diarize on the given audio file using the given model
// and returns speaker segments. The audioPath must be a WAV file on disk.
func Diarize(modelPath, audioPath string) ([]Segment, error) {
	binPath, err := findDiarizer()
	if err != nil {
		return nil, err
	}

	cmd := exec.Command(binPath, modelPath, audioPath)
	cmd.Stderr = os.Stderr

	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("sona-diarize failed: %w", err)
	}

	var segments []Segment
	if err := json.Unmarshal(out, &segments); err != nil {
		return nil, fmt.Errorf("sona-diarize returned invalid JSON: %w", err)
	}

	return segments, nil
}
