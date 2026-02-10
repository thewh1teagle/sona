//go:build linux || darwin || windows

package whisper

/*
#include <whisper.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

static int sona_whisper_verbose = 0;

static void sona_whisper_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (!sona_whisper_verbose) {
        return;
    }
    fputs(text, stderr);
}

static void sona_whisper_set_verbose(int verbose) {
    sona_whisper_verbose = verbose;
    whisper_log_set(sona_whisper_log_callback, NULL);
}

// Forward declarations for Go-exported callback trampolines.
// These are defined in whisper_callbacks_cgo.go via //export.
// CGo generates them as: void sonaGoProgressCB(GoUintptr, GoInt32)
// where GoUintptr = uintptr_t and GoInt32 = int32_t.
extern void sonaGoProgressCB(uintptr_t handle, int32_t progress);
extern void sonaGoSegmentCB(uintptr_t handle, uintptr_t ctx_ptr, int32_t n_new);
extern int32_t sonaGoAbortCB(uintptr_t handle);

// C trampolines that match whisper.h callback signatures.
static void sona_progress_trampoline(struct whisper_context *ctx, struct whisper_state *state, int progress, void *user_data) {
    (void)ctx; (void)state;
    sonaGoProgressCB((uintptr_t)user_data, (int32_t)progress);
}

static void sona_new_segment_trampoline(struct whisper_context *ctx, struct whisper_state *state, int n_new, void *user_data) {
    (void)state;
    sonaGoSegmentCB((uintptr_t)user_data, (uintptr_t)ctx, (int32_t)n_new);
}

static _Bool sona_abort_trampoline(void *user_data) {
    return sonaGoAbortCB((uintptr_t)user_data) != 0;
}

// Helper to set all streaming callbacks on params.
static void sona_set_stream_callbacks(struct whisper_full_params *params, void *handle) {
    params->progress_callback = sona_progress_trampoline;
    params->progress_callback_user_data = handle;
    params->new_segment_callback = sona_new_segment_trampoline;
    params->new_segment_callback_user_data = handle;
    params->abort_callback = sona_abort_trampoline;
    params->abort_callback_user_data = handle;
}
*/
import "C"

import (
	"fmt"
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

func New(modelPath string) (*Context, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	params := C.whisper_context_default_params()
	ctx := C.whisper_init_from_file_with_params(cPath, params)
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

	strategy := C.enum_whisper_sampling_strategy(C.WHISPER_SAMPLING_GREEDY)
	if !opts.SamplingGreedy && opts.BeamSize > 0 {
		strategy = C.enum_whisper_sampling_strategy(C.WHISPER_SAMPLING_BEAM_SEARCH)
	}
	params := C.whisper_full_default_params(strategy)
	params.print_special = C.bool(opts.Verbose)
	params.print_progress = C.bool(opts.Verbose)
	params.print_realtime = C.bool(opts.Verbose)
	params.print_timestamps = C.bool(opts.Verbose)

	if opts.Language != "" {
		cLang := C.CString(opts.Language)
		defer C.free(unsafe.Pointer(cLang))
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
		defer C.free(unsafe.Pointer(cPrompt))
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

	// Set up streaming callbacks if any are provided.
	hasCallbacks := cb.OnProgress != nil || cb.OnSegment != nil || cb.ShouldAbort != nil
	var handle cgo.Handle
	if hasCallbacks {
		handle = cgo.NewHandle(&cb)
		defer handle.Delete()
		C.sona_set_stream_callbacks(&params, unsafe.Pointer(handle))
	}

	ret := C.whisper_full(c.ctx, params, (*C.float)(&samples[0]), C.int(len(samples)))
	if ret != 0 {
		return TranscribeResult{}, fmt.Errorf("whisper: transcription failed with code %d", ret)
	}

	// Collect all segments with timestamps.
	nSegments := int(C.whisper_full_n_segments(c.ctx))
	segments := make([]Segment, nSegments)
	for i := 0; i < nSegments; i++ {
		segments[i] = Segment{
			Start: int64(C.whisper_full_get_segment_t0(c.ctx, C.int(i))),
			End:   int64(C.whisper_full_get_segment_t1(c.ctx, C.int(i))),
			Text:  C.GoString(C.whisper_full_get_segment_text(c.ctx, C.int(i))),
		}
	}

	return TranscribeResult{Segments: segments}, nil
}

func (c *Context) Close() {
	if c.ctx != nil {
		C.whisper_free(c.ctx)
		c.ctx = nil
	}
}
