package server

import (
	"fmt"
	"strings"

	"github.com/thewh1teagle/sona/internal/diarize"
	"github.com/thewh1teagle/sona/internal/whisper"
)

// csToSeconds converts whisper centiseconds (10ms units) to seconds.
func csToSeconds(cs int64) float64 {
	return float64(cs) / 100.0
}

// csToSRTTime converts centiseconds to SRT timestamp format HH:MM:SS,mmm.
func csToSRTTime(cs int64) string {
	ms := cs * 10
	s := ms / 1000
	ms = ms % 1000
	m := s / 60
	s = s % 60
	h := m / 60
	m = m % 60
	return fmt.Sprintf("%02d:%02d:%02d,%03d", h, m, s, ms)
}

// csToVTTTime converts centiseconds to WebVTT timestamp format HH:MM:SS.mmm.
func csToVTTTime(cs int64) string {
	ms := cs * 10
	s := ms / 1000
	ms = ms % 1000
	m := s / 60
	s = s % 60
	h := m / 60
	m = m % 60
	return fmt.Sprintf("%02d:%02d:%02d.%03d", h, m, s, ms)
}

// formatSRT formats segments as SubRip (.srt) subtitles.
func formatSRT(segments []whisper.Segment) string {
	var sb strings.Builder
	for i, seg := range segments {
		if i > 0 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "%d\n%s --> %s\n%s\n",
			i+1,
			csToSRTTime(seg.Start),
			csToSRTTime(seg.End),
			strings.TrimSpace(seg.Text),
		)
	}
	return sb.String()
}

// formatVTT formats segments as WebVTT (.vtt) subtitles.
func formatVTT(segments []whisper.Segment) string {
	var sb strings.Builder
	sb.WriteString("WEBVTT\n\n")
	for i, seg := range segments {
		if i > 0 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "%s --> %s\n%s\n",
			csToVTTTime(seg.Start),
			csToVTTTime(seg.End),
			strings.TrimSpace(seg.Text),
		)
	}
	return sb.String()
}

// verboseSegment is the JSON representation of a segment in verbose_json format.
type verboseSegment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Text    string  `json:"text"`
	Speaker *int    `json:"speaker,omitempty"`
}

// verboseJSON is the response body for response_format=verbose_json.
type verboseJSON struct {
	Text     string           `json:"text"`
	Segments []verboseSegment `json:"segments"`
}

// buildVerboseJSON creates the verbose_json response structure.
// If diarSegments is non-nil, each transcription segment is assigned
// the speaker with maximum time overlap.
func buildVerboseJSON(segments []whisper.Segment, diarSegments []diarize.Segment) verboseJSON {
	text := whisper.TranscribeResult{Segments: segments}.Text()
	vSegs := make([]verboseSegment, len(segments))
	for i, seg := range segments {
		vSegs[i] = verboseSegment{
			Start: csToSeconds(seg.Start),
			End:   csToSeconds(seg.End),
			Text:  seg.Text,
		}
		if diarSegments != nil {
			if sp := matchSpeaker(csToSeconds(seg.Start), csToSeconds(seg.End), diarSegments); sp >= 0 {
				id := sp
				vSegs[i].Speaker = &id
			}
		}
	}
	return verboseJSON{Text: text, Segments: vSegs}
}

// matchSpeaker finds the diarization segment with maximum overlap and
// returns its speaker_id, or -1 if no overlap found.
func matchSpeaker(start, end float64, diarSegments []diarize.Segment) int {
	bestID := -1
	bestOverlap := 0.0
	for _, ds := range diarSegments {
		oStart := start
		if ds.Start > oStart {
			oStart = ds.Start
		}
		oEnd := end
		if ds.End < oEnd {
			oEnd = ds.End
		}
		overlap := oEnd - oStart
		if overlap > bestOverlap {
			bestOverlap = overlap
			bestID = ds.SpeakerID
		}
	}
	return bestID
}
