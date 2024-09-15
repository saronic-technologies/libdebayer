#ifndef DEBAYERC_H
#define DEBAYERC_H

#include "cuda_runtime.h"

#include <stdio.h>
#include <stdint.h>

/* // Number of pixels to pad */
#define SARONIC_DEBAYER_PAD 2

// Kernel block size
#define KERNEL_BLOCK_SIZE 8

#ifdef __cplusplus
extern "C" {
#endif
void debayer_mirror_image(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, uint8_t* input_data);

void debayer_rggb2bgr_malvar2004(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);
void debayer_bggr2bgr_malvar2004(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);

void debayer_rggb2bgr_bilinear(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);

void debayer_rggb2bgr_menon2007(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);
void debayer_bggr2bgr_menon2007(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);

#ifdef __cplusplus
}
#endif
#endif // DEBAYERC_H
