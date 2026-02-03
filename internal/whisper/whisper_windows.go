//go:build windows

package whisper

/*
#cgo CFLAGS: -I${SRCDIR}/../../third_party/include
#cgo LDFLAGS: -L${SRCDIR}/../../third_party/lib
#cgo LDFLAGS: -lwhisper -lggml -lggml-base -lggml-cpu -lggml-vulkan
#cgo LDFLAGS: -lvulkan-1 -lstdc++ -lm -lpthread -lgomp
#cgo LDFLAGS: -static-libstdc++ -static-libgcc -static-libgomp
#include <whisper.h>
#include <stdlib.h>
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

func (c *Context) Transcribe(samples []float32) (string, error) {
	if c.ctx == nil {
		return "", fmt.Errorf("whisper: context is nil")
	}

	params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)

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
