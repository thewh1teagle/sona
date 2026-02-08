package server

import (
	"encoding/json"
	"io"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/thewh1teagle/sona/internal/audio"
	"github.com/thewh1teagle/sona/internal/whisper"
)

// handleHealth always returns 200 — the process is alive.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// handleReady returns 200 if a model is loaded, 503 otherwise.
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	s.mu.Lock()
	loaded := s.ctx != nil
	name := s.modelName
	s.mu.Unlock()

	if !loaded {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "not_ready",
			"message": "no model loaded",
		})
		return
	}
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ready",
		"model":  name,
	})
}

// handleModelLoad loads a model from a path in the JSON body.
func (s *Server) handleModelLoad(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Path string `json:"path"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Path == "" {
		writeError(w, http.StatusBadRequest, "request body must contain {\"path\": \"...\"}")
		return
	}

	if err := s.LoadModel(body.Path); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to load model: "+err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "loaded",
		"model":  s.modelName,
	})
}

// handleModelUnload frees the loaded model.
func (s *Server) handleModelUnload(w http.ResponseWriter, r *http.Request) {
	s.UnloadModel()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "unloaded"})
}

// handleTranscription processes an audio file and returns the result
// in the requested format. Rejects concurrent requests with 429.
func (s *Server) handleTranscription(w http.ResponseWriter, r *http.Request) {
	// Reject if busy (one job at a time).
	if !s.mu.TryLock() {
		writeError(w, http.StatusTooManyRequests, "server is busy with another transcription")
		return
	}
	defer s.mu.Unlock()

	if s.ctx == nil {
		writeError(w, http.StatusServiceUnavailable, "no model loaded")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize)

	file, _, err := r.FormFile("file")
	if err != nil {
		writeError(w, http.StatusBadRequest, "missing or invalid 'file' field: "+err.Error())
		return
	}
	defer file.Close()

	samples, err := audio.ReadWithOptions(file, audio.ReadOptions{
		EnhanceAudio: parseBoolFormValue(r.FormValue("enhance_audio")),
	})
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid audio file: "+err.Error())
		return
	}

	samplingStrategy := r.FormValue("sampling_strategy")
	opts := whisper.TranscribeOptions{
		Language:       r.FormValue("language"),
		DetectLanguage: parseBoolFormValue(r.FormValue("detect_language")),
		Translate:      parseBoolFormValue(r.FormValue("translate")),
		Threads:        parseIntFormValue(r.FormValue("n_threads")),
		Prompt:         r.FormValue("prompt"),
		Verbose:        s.verbose,
		Temperature:    parseFloatFormValue(r.FormValue("temperature")),
		MaxTextCtx:     parseIntFormValue(r.FormValue("max_text_ctx")),
		WordTimestamps: parseBoolFormValue(r.FormValue("word_timestamps")),
		MaxSegmentLen:  parseIntFormValue(r.FormValue("max_segment_len")),
		SamplingGreedy: samplingStrategy != "beam_search",
		BestOf:         parseIntFormValue(r.FormValue("best_of")),
		BeamSize:       parseIntFormValue(r.FormValue("beam_size")),
	}

	responseFormat := r.FormValue("response_format")
	if responseFormat == "" {
		responseFormat = "json"
	}

	stream := parseBoolFormValue(r.FormValue("stream"))

	if stream {
		s.handleStreamingTranscription(w, r, samples, opts)
		return
	}

	// Non-streaming: set up abort on client disconnect.
	var aborted atomic.Bool
	go func() {
		<-r.Context().Done()
		aborted.Store(true)
	}()

	result, err := s.ctx.TranscribeStream(samples, opts, whisper.StreamCallbacks{
		ShouldAbort: func() bool { return aborted.Load() },
	})
	if err != nil {
		if aborted.Load() {
			return // client gone, nothing to write
		}
		writeError(w, http.StatusInternalServerError, "transcription failed: "+err.Error())
		return
	}

	switch responseFormat {
	case "verbose_json":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(buildVerboseJSON(result.Segments))
	case "text":
		w.Header().Set("Content-Type", "text/plain")
		io.WriteString(w, result.Text())
	case "srt":
		w.Header().Set("Content-Type", "text/plain")
		io.WriteString(w, formatSRT(result.Segments))
	case "vtt":
		w.Header().Set("Content-Type", "text/plain")
		io.WriteString(w, formatVTT(result.Segments))
	default: // "json"
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"text": result.Text()})
	}
}

// handleStreamingTranscription writes newline-delimited JSON events
// as segments and progress updates arrive during transcription.
func (s *Server) handleStreamingTranscription(w http.ResponseWriter, r *http.Request, samples []float32, opts whisper.TranscribeOptions) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	enc := json.NewEncoder(w)

	var aborted atomic.Bool
	go func() {
		<-r.Context().Done()
		aborted.Store(true)
	}()

	cb := whisper.StreamCallbacks{
		OnProgress: func(progress int) {
			enc.Encode(map[string]any{
				"type":     "progress",
				"progress": progress,
			})
			flusher.Flush()
		},
		OnSegment: func(seg whisper.Segment) {
			enc.Encode(map[string]any{
				"type":  "segment",
				"start": csToSeconds(seg.Start),
				"end":   csToSeconds(seg.End),
				"text":  seg.Text,
			})
			flusher.Flush()
		},
		ShouldAbort: func() bool { return aborted.Load() },
	}

	result, err := s.ctx.TranscribeStream(samples, opts, cb)
	if err != nil {
		if !aborted.Load() {
			enc.Encode(map[string]any{
				"type":    "error",
				"message": err.Error(),
			})
			flusher.Flush()
		}
		return
	}

	// Final result line.
	enc.Encode(map[string]any{
		"type": "result",
		"text": result.Text(),
	})
	flusher.Flush()
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	name := s.modelName
	loaded := s.ctx != nil
	s.mu.Unlock()

	var data []map[string]any
	if loaded {
		data = []map[string]any{
			{
				"id":       name,
				"object":   "model",
				"created":  time.Now().Unix(),
				"owned_by": "local",
			},
		}
	} else {
		data = []map[string]any{}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   data,
	})
}
