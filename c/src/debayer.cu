#include "debayer.h"

#include "debayer_kernel.h"

void debayer_mirror_image(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, uint8_t* input_data) {
    // Mirror top/bottom edges and corners
    {
        dim3 blockSize(128, SARONIC_DEBAYER_PAD);
        dim3 gridSize(
            (width + SARONIC_DEBAYER_PAD * 2 + blockSize.x - 1) / blockSize.x,
            (SARONIC_DEBAYER_PAD + blockSize.y - 1) / blockSize.y
        );

        mirrorEdgesTopBottom<<<gridSize, blockSize, 0, stream>>>(
            input_data,
            width,
            height,
            input_pitch,
            SARONIC_DEBAYER_PAD
        );
    }

    // Mirror left/right edges
    {
        dim3 blockSize(SARONIC_DEBAYER_PAD, 128);
        dim3 gridSize(
            (SARONIC_DEBAYER_PAD + blockSize.x - 1) / blockSize.x,
            (height + blockSize.y - 1) / blockSize.y
        );

        mirrorEdgesLeftRight<<<gridSize, blockSize, 0, stream>>>(
            input_data,
            width,
            height,
            input_pitch,
            SARONIC_DEBAYER_PAD
        );
    }
}

void debayer_rggb2bgr_malvar2004(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
  dim3 blockSize(KERNEL_BLOCK_SIZE, KERNEL_BLOCK_SIZE);
  dim3 gridSize(
                (width/2  + blockSize.x - 1) / blockSize.x,
                (height/2 + blockSize.y - 1) / blockSize.y
                );
  rggb_malvar2004<<<gridSize, blockSize, 0, stream>>>(
      input_data + SARONIC_DEBAYER_PAD * input_pitch + SARONIC_DEBAYER_PAD,
      input_pitch,
      output_data + SARONIC_DEBAYER_PAD * output_pitch + SARONIC_DEBAYER_PAD * 3,
      output_pitch);
}

void debayer_bggr2bgr_malvar2004(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
  dim3 blockSize(KERNEL_BLOCK_SIZE, KERNEL_BLOCK_SIZE);
  dim3 gridSize(
                (width/2  + blockSize.x - 1) / blockSize.x,
                (height/2 + blockSize.y - 1) / blockSize.y
                );
  bggr_malvar2004<<<gridSize, blockSize, 0, stream>>>(
      input_data + SARONIC_DEBAYER_PAD * input_pitch + SARONIC_DEBAYER_PAD,
      input_pitch,
      output_data + SARONIC_DEBAYER_PAD * output_pitch + SARONIC_DEBAYER_PAD * 3,
      output_pitch);
}

void debayer_rggb2bgr_bilinear(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
  dim3 blockSize(KERNEL_BLOCK_SIZE, KERNEL_BLOCK_SIZE);
  dim3 gridSize(
                (width/2  + blockSize.x - 1) / blockSize.x,
                (height/2 + blockSize.y - 1) / blockSize.y
                );
  rggb_bilinear<<<gridSize, blockSize, 0, stream>>>(
      input_data + SARONIC_DEBAYER_PAD * input_pitch + SARONIC_DEBAYER_PAD,
      input_pitch,
      output_data + SARONIC_DEBAYER_PAD * output_pitch + SARONIC_DEBAYER_PAD * 3,
      output_pitch);
}

void debayer_rggb2bgr_menon2007(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
  dim3 blockSize(KERNEL_BLOCK_SIZE, KERNEL_BLOCK_SIZE);
  dim3 gridSize(
                (width/2  + blockSize.x - 1) / blockSize.x,
                (height/2 + blockSize.y - 1) / blockSize.y
                );
  rggb_menon2007_g<<<gridSize, blockSize, 0, stream>>>(
      input_data + SARONIC_DEBAYER_PAD * input_pitch + SARONIC_DEBAYER_PAD,
      input_pitch,
      output_data + SARONIC_DEBAYER_PAD * output_pitch + SARONIC_DEBAYER_PAD * 3,
      output_pitch);
  rggb_menon2007_rb<<<gridSize, blockSize, 0, stream>>>(
      output_data + SARONIC_DEBAYER_PAD * output_pitch + SARONIC_DEBAYER_PAD * 3,
      output_pitch);
}

void debayer_bggr2bgr_menon2007(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
  dim3 blockSize(KERNEL_BLOCK_SIZE, KERNEL_BLOCK_SIZE);
  dim3 gridSize(
                (width/2  + blockSize.x - 1) / blockSize.x,
                (height/2 + blockSize.y - 1) / blockSize.y
                );
  bggr_menon2007_g<<<gridSize, blockSize, 0, stream>>>(
      input_data + SARONIC_DEBAYER_PAD * input_pitch + SARONIC_DEBAYER_PAD,
      input_pitch,
      output_data + SARONIC_DEBAYER_PAD * output_pitch + SARONIC_DEBAYER_PAD * 3,
      output_pitch);
  bggr_menon2007_rb<<<gridSize, blockSize, 0, stream>>>(
      output_data + SARONIC_DEBAYER_PAD * output_pitch + SARONIC_DEBAYER_PAD * 3,
      output_pitch);
}
