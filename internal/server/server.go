package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"sync"
	"time"

	"github.com/thewh1teagle/sona/internal/audio"
	"github.com/thewh1teagle/sona/internal/whisper"
)

const maxUploadSize = 25 << 20 // 25 MB

type Server struct {
	ctx       *whisper.Context
	mu        sync.Mutex
	modelName string
	verbose   bool
}

func New(ctx *whisper.Context, modelPath string, verbose bool) *Server {
	return &Server{
		ctx:       ctx,
		modelName: filepath.Base(modelPath),
		verbose:   verbose,
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/audio/transcriptions", s.handleTranscription)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	return mux
}

func (s *Server) handleTranscription(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize)

	file, _, err := r.FormFile("file")
	if err != nil {
		writeError(w, http.StatusBadRequest, "missing or invalid 'file' field: "+err.Error())
		return
	}
	defer file.Close()

	samples, err := audio.Read(file)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid audio file: "+err.Error())
		return
	}

	opts := whisper.TranscribeOptions{
		Language: r.FormValue("language"),
		Prompt:   r.FormValue("prompt"),
		Verbose:  s.verbose,
	}

	s.mu.Lock()
	text, err := s.ctx.Transcribe(samples, opts)
	s.mu.Unlock()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "transcription failed: "+err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"text": text})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	now := time.Now().Unix()
	resp := map[string]any{
		"object": "list",
		"data": []map[string]any{
			{
				"id":       s.modelName,
				"object":   "model",
				"created":  now,
				"owned_by": "local",
			},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func writeError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{
			"message": message,
		},
	})
}

func ListenAndServe(addr string, s *Server) error {
	log.Printf("listening on %s", addr)
	fmt.Printf("listening on %s\n", addr)
	return http.ListenAndServe(addr, s.Handler())
}
