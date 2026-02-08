package whisper

import (
	"errors"
	"strings"
)

var ErrNotImplemented = errors.New("whisper: not implemented on this platform")

// TranscribeOptions controls transcription behavior.
type TranscribeOptions struct {
	Language        string  // e.g. "en", "he" (empty = whisper.cpp default: "en")
	DetectLanguage  bool    // auto-detect language (whisper.cpp detect_language)
	Translate       bool    // translate to English
	Threads         int     // CPU threads (0 = whisper default)
	Prompt          string  // initial prompt / vocabulary hint
	Verbose         bool    // enable whisper/ggml logs
	Temperature     float32 // initial decoding temperature (0 = whisper default)
	MaxTextCtx      int     // max tokens from past text as context (0 = whisper default)
	WordTimestamps  bool    // enable token-level timestamps
	MaxSegmentLen   int     // max segment length in characters (0 = no limit)
	SamplingGreedy  bool    // use greedy strategy (default); false = beam search
	BestOf          int     // greedy: number of top candidates (0 = whisper default)
	BeamSize        int     // beam search: beam width (0 = whisper default)
}

// Segment represents a transcribed text segment with timestamps.
type Segment struct {
	Start int64  // start time in centiseconds (10ms units)
	End   int64  // end time in centiseconds (10ms units)
	Text  string
}

// TranscribeResult holds the output of a transcription.
type TranscribeResult struct {
	Segments []Segment
}

// Text returns the concatenated text of all segments.
func (r TranscribeResult) Text() string {
	var sb strings.Builder
	for _, s := range r.Segments {
		sb.WriteString(s.Text)
	}
	return sb.String()
}

// StreamCallbacks provides real-time feedback during transcription.
type StreamCallbacks struct {
	// OnProgress is called with a percentage (0-100) during inference.
	OnProgress func(progress int)
	// OnSegment is called for each newly generated segment.
	OnSegment func(segment Segment)
	// ShouldAbort is polled during inference; return true to cancel.
	ShouldAbort func() bool
}
