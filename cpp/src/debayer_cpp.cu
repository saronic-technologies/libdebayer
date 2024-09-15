#include "debayer_cpp.h"

#include "libdebayer/debayer.h"

#include <cuda_runtime.h>
#include <iostream>

//------------------------------------------------------------------------------
// Algorithm

void Debayer::Free()
{
    cudaFree(raw_cuda_data);
    cudaFree(bgr_cuda_data);

    cudaStreamDestroy(stream);
}

static int RoundUp(int x, int modulus) {
    int r = x % modulus;
    if (r == 0) {
        return x;
    } else {
    return x + (modulus - r);
    }
}

bool Debayer::Allocate(int width, int height)
{
    if (cuda_width == width && cuda_height == height) {
        return true;
    }

    cuda_width = width;
    cuda_height = height;

    Free();

    cudaStreamCreate(&stream);

    cudaError_t r;

    // Pixel-sized buffers (one value per pixel)
    {
        const int padded_width = SARONIC_DEBAYER_PAD + RoundUp(width + SARONIC_DEBAYER_PAD, KERNEL_BLOCK_SIZE);
        const int padded_height = SARONIC_DEBAYER_PAD + RoundUp(height + SARONIC_DEBAYER_PAD, KERNEL_BLOCK_SIZE);

        r = cudaMallocPitch(&raw_cuda_data, &raw_cuda_pitch, padded_width, padded_height);
        if (r != cudaSuccess) { return false; }
        r = cudaMemset2D(raw_cuda_data, raw_cuda_pitch, 0, padded_width, padded_height);
        if (r != cudaSuccess) { return false; }

        r = cudaMallocPitch(&bgr_cuda_data, &bgr_cuda_pitch, padded_width * 3, padded_height);
        if (r != cudaSuccess) { return false; }
        r = cudaMemset2D(bgr_cuda_data, bgr_cuda_pitch, 0, padded_width * 3, padded_height);
        if (r != cudaSuccess) { return false; }
    }

    return true;
}

int32_t Debayer::Process(const raw_image_t* input, const bgr_image_t* output)
{
    if (input->width != output->width || input->height != output->height) {
        std::cerr << "Error: input and output image sizes do not match" << std::endl;
        return -1;
    }

    if (!Allocate(input->width, input->height)) {
        Free();
        return -2;
    }

    cudaError_t r = cudaMemcpy2DAsync(
        raw_cuda_data + SARONIC_DEBAYER_PAD * raw_cuda_pitch + SARONIC_DEBAYER_PAD,
        raw_cuda_pitch,
        input->raw_data,
        input->pitch != 0 ? input->pitch : input->width,
        input->width,
        input->height,
        cudaMemcpyHostToDevice,
        stream);
    if (r != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy2DAsync(cpu->gpu) failed: " << cudaGetErrorString(r) << std::endl;
        return -3;
    }

    debayer_mirror_image(stream, input->width, input->height, raw_cuda_pitch, raw_cuda_data);


    if (input->algorithm == SARONIC_DEBAYER_BILINEAR) {
      debayer_rggb2bgr_bilinear(stream, input->width, input->height, raw_cuda_pitch, bgr_cuda_pitch, raw_cuda_data, bgr_cuda_data);
    } else if (input->algorithm == SARONIC_DEBAYER_MALVAR2004) {
        if (input->format == SARONIC_DEBAYER_RGGB) {
          debayer_rggb2bgr_malvar2004(stream, input->width, input->height, raw_cuda_pitch, bgr_cuda_pitch, raw_cuda_data, bgr_cuda_data);
        } else if (input->format == SARONIC_DEBAYER_BGGR) {
          debayer_bggr2bgr_malvar2004(stream, input->width, input->height, raw_cuda_pitch, bgr_cuda_pitch, raw_cuda_data, bgr_cuda_data);
        }
    } else if (input->algorithm == SARONIC_DEBAYER_MENON2007) {
        if (input->format == SARONIC_DEBAYER_RGGB) {
          debayer_rggb2bgr_menon2007(stream, input->width, input->height, raw_cuda_pitch, bgr_cuda_pitch, raw_cuda_data, bgr_cuda_data);
        } else if (input->format == SARONIC_DEBAYER_BGGR) {
          debayer_bggr2bgr_menon2007(stream, input->width, input->height, raw_cuda_pitch, bgr_cuda_pitch, raw_cuda_data, bgr_cuda_data);
        }
    } else {
        std::cerr << "Error: unknown algorithm: " << input->algorithm << std::endl;
        return -6;
    }

    r = cudaMemcpy2DAsync(
        output->bgr_data,
        output->pitch != 0 ? output->pitch : output->width * 3,
        bgr_cuda_data + SARONIC_DEBAYER_PAD * bgr_cuda_pitch + SARONIC_DEBAYER_PAD * 3,
        bgr_cuda_pitch,
        output->width * 3,
        output->height,
        cudaMemcpyDeviceToHost,
        stream);
    if (r != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy2DAsync(gpu->cpu) failed: " << cudaGetErrorString(r) << std::endl;
        return -4;
    }

    r = cudaStreamSynchronize(stream);
    if (r != cudaSuccess) {
        std::cerr << "Error: cudaStreamSynchronize failed: " << cudaGetErrorString(r) << std::endl;
        return -5;
    }

    return 0;
}
