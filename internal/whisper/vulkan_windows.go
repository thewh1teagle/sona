//go:build windows

package whisper

import "golang.org/x/sys/windows"

// VulkanAvailable reports whether vulkan-1.dll can be loaded on this system.
func VulkanAvailable() bool {
	dll, err := windows.LoadDLL("vulkan-1.dll")
	if err != nil {
		return false
	}
	dll.Release()
	return true
}
