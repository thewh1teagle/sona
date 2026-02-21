//go:build linux || darwin || windows

package whisper

import (
	"fmt"
	"testing"
)

func TestListGPUDevices(t *testing.T) {
	devices := ListGPUDevices()
	fmt.Printf("Found %d GPU device(s):\n", len(devices))
	for _, d := range devices {
		fmt.Printf("  [%d] %s - %s (type: %s)\n", d.Index, d.Name, d.Description, d.Type)
	}
}
