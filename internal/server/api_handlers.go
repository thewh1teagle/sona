package server

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"sync/atomic"
	"time"

	"github.com/thewh1teagle/sona/internal/audio"
	"github.com/thewh1teagle/sona/internal/diarize"
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
		Path      string `json:"path"`
		GpuDevice *int   `json:"gpu_device,omitempty"` // optional; nil = whisper default
		NoGpu     bool   `json:"no_gpu,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Path == "" {
		writeError(w, http.StatusBadRequest, "request body must contain {\"path\": \"...\"}")
		return
	}

	gpuDevice := -1
	if body.GpuDevice != nil {
		gpuDevice = *body.GpuDevice
	}

	if err := s.LoadModel(body.Path, gpuDevice, body.NoGpu); err != nil {
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

	diarizeModel := r.FormValue("diarize_model")

	// If diarization requested, save upload to temp file, then convert to
	// native 16kHz mono PCM WAV so sona-diarize can read it. The converted
	// file is also used for whisper (skips its own ffmpeg pass).
	var tempAudioPath string
	var fileReader io.ReadSeeker = file
	if diarizeModel != "" {
		tmp, tmpErr := os.CreateTemp("", "sona-diar-*.audio")
		if tmpErr != nil {
			writeError(w, http.StatusInternalServerError, "failed to create temp file: "+tmpErr.Error())
			return
		}
		defer os.Remove(tmp.Name())
		if _, copyErr := io.Copy(tmp, file); copyErr != nil {
			tmp.Close()
			writeError(w, http.StatusInternalServerError, "failed to buffer upload: "+copyErr.Error())
			return
		}
		tmp.Close()

		// Convert to native WAV for diarization (and reuse for whisper).
		nativeWav := tmp.Name() + ".wav"
		if convErr := audio.ConvertToNativeWav(tmp.Name(), nativeWav, false); convErr != nil {
			log.Printf("failed to convert audio to native WAV: %v", convErr)
			writeError(w, http.StatusBadRequest, "failed to convert audio for diarization: "+convErr.Error())
			return
		}
		defer os.Remove(nativeWav)
		tempAudioPath = nativeWav

		// Reopen converted file for audio decoding
		reopened, reopenErr := os.Open(nativeWav)
		if reopenErr != nil {
			writeError(w, http.StatusInternalServerError, "failed to reopen converted file: "+reopenErr.Error())
			return
		}
		defer reopened.Close()
		fileReader = reopened
	}

	samples, err := audio.ReadWithOptions(fileReader, audio.ReadOptions{
		EnhanceAudio: parseBoolFormValue(r.FormValue("enhance_audio")),
	})
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid audio file: "+err.Error())
		return
	}

	// Start diarization in background if requested.
	type diarResult struct {
		segments []diarize.Segment
		err      error
	}
	var diarCh chan diarResult
	if diarizeModel != "" && tempAudioPath != "" {
		diarCh = make(chan diarResult, 1)
		go func() {
			segs, dErr := diarize.Diarize(diarizeModel, tempAudioPath)
			diarCh <- diarResult{segs, dErr}
		}()
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
		// Run diarization before streaming so speaker labels are available for each segment.
		var diarStreamSegments []diarize.Segment
		if diarizeModel != "" && tempAudioPath != "" {
			segs, dErr := diarize.Diarize(diarizeModel, tempAudioPath)
			if dErr != nil {
				log.Printf("diarization failed (streaming without speakers): %v", dErr)
			} else {
				diarStreamSegments = segs
			}
		}
		s.handleStreamingTranscription(w, r, samples, opts, diarStreamSegments)
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

	// Collect diarization results (skip silently on failure).
	var diarSegments []diarize.Segment
	if diarCh != nil {
		dr := <-diarCh
		if dr.err != nil {
			log.Printf("diarization failed (skipping): %v", dr.err)
		} else {
			diarSegments = dr.segments
		}
	}

	switch responseFormat {
	case "verbose_json":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(buildVerboseJSON(result.Segments, diarSegments))
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
func (s *Server) handleStreamingTranscription(w http.ResponseWriter, r *http.Request, samples []float32, opts whisper.TranscribeOptions, diarSegments []diarize.Segment) {
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
			event := map[string]any{
				"type":  "segment",
				"start": csToSeconds(seg.Start),
				"end":   csToSeconds(seg.End),
				"text":  seg.Text,
			}
			if diarSegments != nil {
				if sp := matchSpeaker(csToSeconds(seg.Start), csToSeconds(seg.End), diarSegments); sp >= 0 {
					event["speaker"] = sp
				}
			}
			enc.Encode(event)
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

// handleDevices returns GPU/iGPU devices discovered by ggml backends.
func (s *Server) handleDevices(w http.ResponseWriter, r *http.Request) {
	devices := whisper.ListGPUDevices()
	if devices == nil {
		devices = []whisper.GPUDevice{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   devices,
	})
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
