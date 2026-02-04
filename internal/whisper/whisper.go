package whisper

import "errors"

var ErrNotImplemented = errors.New("whisper: not implemented on this platform")

// TranscribeOptions controls transcription behavior.
type TranscribeOptions struct {
	Language  string // e.g. "en", "he", "auto" (empty = auto)
	Translate bool   // translate to English
	Threads   int    // CPU threads (0 = whisper default)
	Prompt    string // initial prompt / vocabulary hint
}
