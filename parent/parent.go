package parent

import (
	"os"
)

var parentPID = os.Getppid()

// ParentExited returns true if the original parent process is gone.
func ParentExited() bool {
	return !processExists(parentPID)
}
