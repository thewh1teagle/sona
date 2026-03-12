//go:build windows

package parent

import (
	"syscall"
)

func processExists(pid int) bool {
	handle, err := syscall.OpenProcess(syscall.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return false
	}
	defer syscall.CloseHandle(handle)

	var code uint32
	err = syscall.GetExitCodeProcess(handle, &code)
	if err != nil {
		return false
	}

	const STILL_ACTIVE = 259
	return code == STILL_ACTIVE
}
