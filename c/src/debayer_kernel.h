#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// CUDA Kernels

extern __global__ void mirrorEdgesTopBottom(
    uint8_t* data,
    int width,
    int height,
    int pitch,
    int pad);

extern __global__ void mirrorEdgesLeftRight(
    uint8_t* data,
    int width,
    int height,
    int pitch,
    int pad);

extern __global__ void rggb_malvar2004(
    const uint8_t* raw,
    size_t raw_pitch,
    uint8_t* bgr,
    size_t bgr_pitch);

extern __global__ void bggr_malvar2004(
    const uint8_t* raw,
    size_t raw_pitch,
    uint8_t* bgr,
    size_t bgr_pitch);

extern __global__ void rggb_bilinear(
    const uint8_t* raw,
    size_t raw_pitch,
    uint8_t* bgr,
    size_t bgr_pitch);

extern __global__ void rggb_menon2007_g(
    const uint8_t* raw,
    int raw_pitch,
    uint8_t* bgr,
    int bgr_pitch);

extern __global__ void rggb_menon2007_rb(
    uint8_t* bgr,
    int bgr_pitch);

extern __global__ void bggr_menon2007_g(
    const uint8_t* raw,
    int raw_pitch,
    uint8_t* bgr,
    int bgr_pitch);

extern __global__ void bggr_menon2007_rb(
    uint8_t* bgr,
    int bgr_pitch);
