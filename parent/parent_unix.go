//go:build linux || darwin

package parent

import "syscall"

func processExists(pid int) bool {
	err := syscall.Kill(pid, 0)
	return err == nil
}
