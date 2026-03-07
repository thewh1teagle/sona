//go:build linux || darwin || windows

package whisper

/*
#include "whisper_cgo.h"
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"fmt"
	"os"
	"runtime/cgo"
	"unsafe"
)

type Context struct {
	ctx *C.struct_whisper_context
}

func SetVerbose(v bool) {
	if v {
		C.sona_whisper_set_verbose(1)
		return
	}
	C.sona_whisper_set_verbose(0)
}

func New(modelPath string, gpuDevice int, noGpu bool) (*Context, error) {
	// Read model via Go's os.ReadFile which handles non-ASCII paths on Windows
	// (Go uses CreateFileW internally), then pass the buffer to whisper.cpp
	// to avoid fopen() failing on non-ASCII paths with MinGW's C runtime.
	buf, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("whisper: failed to read model file %s: %w", modelPath, err)
	}
	if len(buf) == 0 {
		return nil, fmt.Errorf("whisper: model file is empty or corrupt: %s", modelPath)
	}

	params := C.whisper_context_default_params()
	if noGpu || !VulkanAvailable() {
		params.use_gpu = C.bool(false)
	} else if gpuDevice >= 0 {
		params.gpu_device = C.int(gpuDevice)
	}
	ctx := C.whisper_init_from_buffer_with_params(unsafe.Pointer(&buf[0]), C.size_t(len(buf)), params)
	if ctx == nil {
		return nil, fmt.Errorf("whisper: failed to load model from %s", modelPath)
	}
	return &Context{ctx: ctx}, nil
}

// Transcribe runs inference and returns all segments with timestamps.
func (c *Context) Transcribe(samples []float32, opts TranscribeOptions) (TranscribeResult, error) {
	return c.TranscribeStream(samples, opts, StreamCallbacks{})
}

// TranscribeStream runs inference with real-time callbacks for progress, segments, and cancellation.
func (c *Context) TranscribeStream(samples []float32, opts TranscribeOptions, cb StreamCallbacks) (TranscribeResult, error) {
	if c.ctx == nil {
		return TranscribeResult{}, fmt.Errorf("whisper: context is nil")
	}
	if len(samples) == 0 {
		return TranscribeResult{}, fmt.Errorf("whisper: no samples")
	}
	if opts.StableTimestamps {
		return c.transcribeStableTimestamps(samples, opts, cb)
	}

	params, cleanup := buildFullParams(opts)
	defer cleanup()

	// Set up streaming callbacks if any are provided.
	hasCallbacks := cb.OnProgress != nil || cb.OnSegment != nil || cb.ShouldAbort != nil
	var handle cgo.Handle
	if hasCallbacks {
		handle = cgo.NewHandle(&cb)
		defer handle.Delete()
		C.sona_whisper_set_stream_callbacks(&params, C.uintptr_t(handle))
	}

	ret := C.whisper_full(c.ctx, params, (*C.float)(&samples[0]), C.int(len(samples)))
	if ret != 0 {
		return TranscribeResult{}, fmt.Errorf("whisper: transcription failed with code %d", ret)
	}

	return TranscribeResult{Segments: collectSegments(c.ctx)}, nil
}

func buildFullParams(opts TranscribeOptions) (C.struct_whisper_full_params, func()) {
	strategy := C.enum_whisper_sampling_strategy(C.WHISPER_SAMPLING_GREEDY)
	if !opts.SamplingGreedy && opts.BeamSize > 0 {
		strategy = C.enum_whisper_sampling_strategy(C.WHISPER_SAMPLING_BEAM_SEARCH)
	}
	params := C.whisper_full_default_params(strategy)
	params.print_special = C.bool(opts.Verbose)
	params.print_progress = C.bool(opts.Verbose)
	params.print_realtime = C.bool(opts.Verbose)
	params.print_timestamps = C.bool(opts.Verbose)

	var cPtrs []unsafe.Pointer
	if opts.Language != "" {
		cLang := C.CString(opts.Language)
		cPtrs = append(cPtrs, unsafe.Pointer(cLang))
		params.language = cLang // whisper.cpp handles "auto" natively
	}
	if opts.DetectLanguage {
		params.detect_language = C.bool(true)
	}
	if opts.Translate {
		params.translate = C.bool(true)
	}
	if opts.Threads > 0 {
		params.n_threads = C.int(opts.Threads)
	}
	if opts.Prompt != "" {
		cPrompt := C.CString(opts.Prompt)
		cPtrs = append(cPtrs, unsafe.Pointer(cPrompt))
		params.initial_prompt = cPrompt
	}
	if opts.Temperature > 0 {
		params.temperature = C.float(opts.Temperature)
	}
	if opts.MaxTextCtx > 0 {
		params.n_max_text_ctx = C.int(opts.MaxTextCtx)
	}
	if opts.WordTimestamps {
		params.token_timestamps = C.bool(true)
	}
	if opts.MaxSegmentLen > 0 {
		params.max_len = C.int(opts.MaxSegmentLen)
	}
	if opts.BestOf > 0 {
		params.greedy.best_of = C.int(opts.BestOf)
	}
	if opts.BeamSize > 0 {
		params.beam_search.beam_size = C.int(opts.BeamSize)
	}

	cleanup := func() {
		for _, ptr := range cPtrs {
			C.free(ptr)
		}
	}
	return params, cleanup
}

func collectSegments(ctx *C.struct_whisper_context) []Segment {
	nSegments := int(C.whisper_full_n_segments(ctx))
	segments := make([]Segment, nSegments)
	for i := 0; i < nSegments; i++ {
		segments[i] = Segment{
			Start: int64(C.whisper_full_get_segment_t0(ctx, C.int(i))),
			End:   int64(C.whisper_full_get_segment_t1(ctx, C.int(i))),
			Text:  C.GoString(C.whisper_full_get_segment_text(ctx, C.int(i))),
		}
	}
	return segments
}

func (c *Context) transcribeStableTimestamps(samples []float32, opts TranscribeOptions, cb StreamCallbacks) (TranscribeResult, error) {
	if opts.VadModelPath == "" {
		return TranscribeResult{}, fmt.Errorf("whisper: vad_model is required when stable timestamps are enabled")
	}

	params, cleanup := buildFullParams(opts)
	defer cleanup()
	params.vad = C.bool(false)
	params.vad_model_path = nil

	// Keep abort support active during segment decode without emitting raw callbacks.
	if cb.ShouldAbort != nil {
		abortOnly := StreamCallbacks{ShouldAbort: cb.ShouldAbort}
		handle := cgo.NewHandle(&abortOnly)
		defer handle.Delete()
		C.sona_whisper_set_stream_callbacks(&params, C.uintptr_t(handle))
	}

	cVadModelPath := C.CString(opts.VadModelPath)
	defer C.free(unsafe.Pointer(cVadModelPath))

	vadCtxParams := C.whisper_vad_default_context_params()
	vctx := C.whisper_vad_init_from_file_with_params(cVadModelPath, vadCtxParams)
	if vctx == nil {
		return TranscribeResult{}, fmt.Errorf("whisper: failed to load VAD model from %s", opts.VadModelPath)
	}
	defer C.whisper_vad_free(vctx)

	vadParams := C.whisper_vad_default_params()
	vadSegments := C.whisper_vad_segments_from_samples(vctx, vadParams, (*C.float)(&samples[0]), C.int(len(samples)))
	if vadSegments == nil {
		return TranscribeResult{}, fmt.Errorf("whisper: failed to run VAD segmentation")
	}
	defer C.whisper_vad_free_segments(vadSegments)

	nVadSegments := int(C.whisper_vad_segments_n_segments(vadSegments))
	if nVadSegments == 0 {
		if cb.OnProgress != nil {
			cb.OnProgress(100)
		}
		return TranscribeResult{Segments: []Segment{}}, nil
	}

	result := TranscribeResult{Segments: make([]Segment, 0, nVadSegments)}
	for i := 0; i < nVadSegments; i++ {
		if cb.ShouldAbort != nil && cb.ShouldAbort() {
			return TranscribeResult{}, fmt.Errorf("whisper: transcription aborted")
		}

		t0cs := int64(C.whisper_vad_segments_get_segment_t0(vadSegments, C.int(i)))
		t1cs := int64(C.whisper_vad_segments_get_segment_t1(vadSegments, C.int(i)))
		if t1cs <= t0cs {
			continue
		}

		start := int(float64(t0cs) * float64(C.WHISPER_SAMPLE_RATE) / 100.0)
		end := int(float64(t1cs) * float64(C.WHISPER_SAMPLE_RATE) / 100.0)
		if start < 0 {
			start = 0
		}
		if end > len(samples) {
			end = len(samples)
		}
		if end <= start {
			continue
		}

		ret := C.whisper_full(c.ctx, params, (*C.float)(&samples[start]), C.int(end-start))
		if ret != 0 {
			return TranscribeResult{}, fmt.Errorf("whisper: transcription failed with code %d", ret)
		}

		decoded := collectSegments(c.ctx)
		for _, seg := range decoded {
			shifted := seg
			shifted.Start += t0cs
			shifted.End += t0cs
			result.Segments = append(result.Segments, shifted)
			if cb.OnSegment != nil {
				cb.OnSegment(shifted)
			}
		}

		if cb.OnProgress != nil {
			cb.OnProgress((i + 1) * 100 / nVadSegments)
		}
	}

	return result, nil
}

func (c *Context) Close() {
	if c.ctx != nil {
		C.whisper_free(c.ctx)
		c.ctx = nil
	}
}

// GPUDevice describes a backend device reported by ggml.
type GPUDevice struct {
	Index       int    `json:"index"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Type        string `json:"type"` // "gpu", "igpu", "cpu", "accel"
}

// ListGPUDevices returns GPU and integrated-GPU devices from ggml backends.
func ListGPUDevices() []GPUDevice {
	n := int(C.sona_gpu_device_count())
	var devices []GPUDevice
	for i := 0; i < n; i++ {
		devType := int(C.sona_gpu_device_type(C.int(i)))
		typeName := ""
		switch devType {
		case 1: // GGML_BACKEND_DEVICE_TYPE_GPU
			typeName = "gpu"
		case 2: // GGML_BACKEND_DEVICE_TYPE_IGPU
			typeName = "igpu"
		default:
			continue // skip CPU and ACCEL devices
		}
		devices = append(devices, GPUDevice{
			Index:       i,
			Name:        C.GoString(C.sona_gpu_device_name(C.int(i))),
			Description: C.GoString(C.sona_gpu_device_description(C.int(i))),
			Type:        typeName,
		})
	}
	return devices
}
