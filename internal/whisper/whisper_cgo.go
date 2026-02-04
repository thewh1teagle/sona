//go:build linux || darwin || windows

package whisper

/*
#include <whisper.h>
#include <stdlib.h>
#include <stdio.h>

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
*/
import "C"

import (
	"fmt"
	"strings"
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

func (c *Context) Transcribe(samples []float32, opts TranscribeOptions) (string, error) {
	if c.ctx == nil {
		return "", fmt.Errorf("whisper: context is nil")
	}

	params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
	params.print_special = C.bool(opts.Verbose)
	params.print_progress = C.bool(opts.Verbose)
	params.print_realtime = C.bool(opts.Verbose)
	params.print_timestamps = C.bool(opts.Verbose)

	if opts.Language != "" {
		cLang := C.CString(opts.Language)
		defer C.free(unsafe.Pointer(cLang))
		params.language = cLang
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

	ret := C.whisper_full(c.ctx, params, (*C.float)(&samples[0]), C.int(len(samples)))
	if ret != 0 {
		return "", fmt.Errorf("whisper: transcription failed with code %d", ret)
	}

	nSegments := int(C.whisper_full_n_segments(c.ctx))
	var sb strings.Builder
	for i := 0; i < nSegments; i++ {
		text := C.GoString(C.whisper_full_get_segment_text(c.ctx, C.int(i)))
		sb.WriteString(text)
	}
	return sb.String(), nil
}

func (c *Context) Close() {
	if c.ctx != nil {
		C.whisper_free(c.ctx)
		c.ctx = nil
	}
}
