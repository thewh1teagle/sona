#pragma once
#include <whisper.h>
#include <ggml-backend.h>
#include <stdint.h>

void sona_whisper_set_verbose(int verbose);
void sona_whisper_set_stream_callbacks(struct whisper_full_params *params, uintptr_t handle);

// GPU device enumeration via ggml backend API.
int sona_gpu_device_count(void);
const char *sona_gpu_device_name(int index);
const char *sona_gpu_device_description(int index);
int sona_gpu_device_type(int index);
