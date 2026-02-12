//go:build !windows

package whisper

// VulkanAvailable always returns true on non-Windows platforms.
// Vulkan availability issues only affect Windows (missing vulkan-1.dll).
func VulkanAvailable() bool {
	return true
}
