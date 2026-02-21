package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/thewh1teagle/sona/internal/whisper"
)

const maxUploadSize = 15 << 30 // 15 GB

type Server struct {
	mu        sync.Mutex
	ctx       *whisper.Context // nil when no model loaded
	modelName string
	modelPath string
	verbose   bool
	Version   string
	Commit    string
}

func New(verbose bool) *Server {
	return &Server{verbose: verbose}
}

// LoadModel loads a whisper model, unloading any existing one first.
// gpuDevice selects the GPU (-1 = use whisper default).
func (s *Server) LoadModel(path string, gpuDevice int, noGpu bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loadModelLocked(path, gpuDevice, noGpu)
}

func (s *Server) loadModelLocked(path string, gpuDevice int, noGpu bool) error {
	if s.ctx != nil {
		s.ctx.Close()
		s.ctx = nil
		s.modelName = ""
		s.modelPath = ""
	}

	ctx, err := whisper.New(path, gpuDevice, noGpu)
	if err != nil {
		return err
	}
	s.ctx = ctx
	s.modelPath = path
	s.modelName = filepath.Base(path)
	return nil
}

// UnloadModel frees the current model. Safe to call with no model loaded.
func (s *Server) UnloadModel() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx != nil {
		s.ctx.Close()
		s.ctx = nil
		s.modelName = ""
		s.modelPath = ""
	}
}

// Close frees all resources.
func (s *Server) Close() {
	s.UnloadModel()
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /ready", s.handleReady)
	mux.HandleFunc("POST /v1/models/load", s.handleModelLoad)
	mux.HandleFunc("DELETE /v1/models", s.handleModelUnload)
	mux.HandleFunc("POST /v1/audio/transcriptions", s.handleTranscription)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /v1/devices", s.handleDevices)
	s.registerDocsRoutes(mux)
	return mux
}

// ListenAndServe binds to the given port (0 = auto-assign), prints a ready
// signal to stdout, and serves until interrupted.
func ListenAndServe(host string, port int, s *Server) error {
	ln, err := net.Listen("tcp", fmt.Sprintf("%s:%d", host, port))
	if err != nil {
		return err
	}

	actualPort := ln.Addr().(*net.TCPAddr).Port

	// Machine-readable ready signal for parent process.
	readyMsg, _ := json.Marshal(map[string]any{
		"status":  "ready",
		"port":    actualPort,
		"version": s.Version,
		"commit":  s.Commit,
	})
	fmt.Println(string(readyMsg))
	log.Printf("listening on %s:%d", host, actualPort)

	srv := &http.Server{Handler: s.Handler()}

	// Graceful shutdown on SIGINT/SIGTERM.
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		<-sigCh
		log.Println("shutting down...")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		srv.Shutdown(ctx)
		s.Close()
	}()

	err = srv.Serve(ln)
	if err == http.ErrServerClosed {
		return nil
	}
	return err
}
