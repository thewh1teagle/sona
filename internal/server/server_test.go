package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHealthEndpoint(t *testing.T) {
	s := New(false)
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	s.handleHealth(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var body map[string]string
	json.NewDecoder(w.Body).Decode(&body)
	if body["status"] != "ok" {
		t.Errorf("expected status ok, got %q", body["status"])
	}
}

func TestReadyNoModel(t *testing.T) {
	s := New(false)
	req := httptest.NewRequest("GET", "/ready", nil)
	w := httptest.NewRecorder()
	s.handleReady(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
	var body map[string]string
	json.NewDecoder(w.Body).Decode(&body)
	if body["status"] != "not_ready" {
		t.Errorf("expected status not_ready, got %q", body["status"])
	}
}

func TestModelsEmpty(t *testing.T) {
	s := New(false)
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	s.handleModels(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var body map[string]any
	json.NewDecoder(w.Body).Decode(&body)
	data := body["data"].([]any)
	if len(data) != 0 {
		t.Errorf("expected empty data, got %d items", len(data))
	}
}

func TestTranscriptionNoModel(t *testing.T) {
	s := New(false)
	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", nil)
	w := httptest.NewRecorder()
	s.handleTranscription(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}

func TestModelUnloadIdempotent(t *testing.T) {
	s := New(false)
	req := httptest.NewRequest("DELETE", "/v1/models", nil)
	w := httptest.NewRecorder()
	s.handleModelUnload(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var body map[string]string
	json.NewDecoder(w.Body).Decode(&body)
	if body["status"] != "unloaded" {
		t.Errorf("expected status unloaded, got %q", body["status"])
	}
}
