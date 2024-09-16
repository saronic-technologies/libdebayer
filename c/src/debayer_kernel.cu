#include "debayer_kernel.h"

/*
    References:

    [1] "HIGH-QUALITY LINEAR INTERPOLATION FOR DEMOSAICING OF BAYER-PATTERNED COLOR IMAGES" (Malvar 2004)
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Demosaicing_ICASSP04.pdf

    [2] "Demosaicing With Directional Filtering and a posteriori Decision" (Menon 2007)
        https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=8c8e4a3cf6d0b8dfdcd36652718ad54afd2fe5fe
*/

#include <stdio.h>


//------------------------------------------------------------------------------
// Tools

#define BLUE  0
#define GREEN 1
#define RED   2

#define B_SET_CENTER(x, y) \
    const int V = raw_pitch; \
    const uint8_t* P = block + y * V + x;

#define B_AT(x, y) (int16_t)P[y * V + x]

__device__ inline uint8_t saturate_cast_int16_to_uint8(int16_t val) {
    return static_cast<uint8_t>(max(0, min(255, val)));
}

__device__ inline void WriteBGRBlockPixel(
    uint8_t* bgr_block,
    size_t bgr_pitch,
    int quad_x,
    int quad_y,
    int16_t b,
    int16_t g,
    int16_t r)
{
    uint8_t* bgr_pixel = bgr_block + quad_y * bgr_pitch + quad_x * 3;
    bgr_pixel[0] = saturate_cast_int16_to_uint8(b);
    bgr_pixel[1] = saturate_cast_int16_to_uint8(g);
    bgr_pixel[2] = saturate_cast_int16_to_uint8(r);
}


//------------------------------------------------------------------------------
// Mirror Edges Kernels

__device__ inline int mirror_x(int x, int width) {
    if (x < 0) {
        return 0;
    } else if (x >= width) {
        return width - 1;
    } else {
        return x;
    }
}

__global__ void mirrorEdgesTopBottom(uint8_t* data, int width, int height, int pitch, int pad)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint8_t* image = data + pitch * pad + pad;

#if 0
    // Set padding pixels at top
    uint8_t* border_top = data + y * pitch + x;
    border_top[0] = image[y * pitch + mirror_x(x - pad, width)];

    // Set padding pixels on bottom
    uint8_t* border_bottom = data + (pad + height + y) * pitch + x;
    border_bottom[0] = image[(height - 2 - y) * pitch + mirror_x(x - pad, width)];
#else
    // Set padding pixels at top
    uint8_t* border_top = data + y * pitch + x;
    border_top[0] = image[mirror_x(x - pad, width)];

    // Set padding pixels on bottom
    uint8_t* border_bottom = data + (pad + height + y) * pitch + x;
    border_bottom[0] = image[(height - 1) * pitch + mirror_x(x - pad, width)];
#endif
}

__global__ void mirrorEdgesLeftRight(uint8_t* data, int width, int height, int pitch, int pad)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint8_t* image = data + pitch * pad + pad;

#if 0
    // Set padding pixels at left
    data[(y + pad) * pitch + x] = image[y * pitch + (pad - x)];

    // Set padding pixels at right
    data[(y + pad) * pitch + pad + width + x] = image[y * pitch + (width - 2 - x)];
#else
    // Set padding pixels at left
    data[(y + pad) * pitch + x] = image[y * pitch];

    // Set padding pixels at right
    data[(y + pad) * pitch + pad + width + x] = image[y * pitch + (width - 1)];
#endif
}


//------------------------------------------------------------------------------
// Malvar 2004 Algorithm

__global__ void rggb_malvar2004(
    const uint8_t* raw,
    size_t raw_pitch,
    uint8_t* bgr,
    size_t bgr_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // We are looking at a 2x2 block
    const uint8_t* block = reinterpret_cast<const uint8_t*>(raw + (y * raw_pitch + x) * 2);
    uint8_t* bgr_block = reinterpret_cast<uint8_t*>(bgr + (y * bgr_pitch + x * 3) * 2);

    /*
        RGGB layout:

            R G
            G B

        G at R/B:

            0  0 -1  0  0
            0  0  2  0  0
           -1  2  4  2 -1 + 4) / 8
            0  0  2  0  0
            0  0 -1  0  0

        R at G, R column:

            0  0 -2  0  0
            0 -2  8 -2  0
            1  0 10  0  1 + 8) / 16
            0 -2  8 -2  0
            0  0 -2  0  0

        R at G, B column:

            0  0  1  0  0
            0 -2  0 -2  0
           -2  8 10  8 -2 + 8) / 16
            0 -2  0 -2  0
            0  0  1  0  0

        B at G, B column:

            0  0 -2  0  0
            0 -2  8 -2  0
            1  0 10  0  1 + 8) / 16
            0 -2  8 -2  0
            0  0 -2  0  0

        B at G, R column:

            0  0  1  0  0
            0 -2  0 -2  0
           -2  8 10  8 -2 + 8) / 16
            0 -2  0 -2  0
            0  0  1  0  0

        R at B, B column:

            0  0 -3  0  0
            0  4  0  4  0
           -3  0 12  0 -3 + 8) / 16
            0  4  0  4  0
            0  0 -3  0  0

        B at R, R column:

            0  0 -3  0  0
            0  4  0  4  0
           -3  0 12  0 -3 + 8) / 16
            0  4  0  4  0
            0  0 -3  0  0
    */

    // Upper left:
    {
        B_SET_CENTER(0,0);
        int16_t b = (12 * B_AT(0,0)
                   + 4 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1))
                   - 3 * (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 8) / 16;
        int16_t g = (4 * B_AT(0,0)
                   + 2 * (B_AT(0,-1) + B_AT(0,1) + B_AT(-1,0) + B_AT(1,0))
                   - (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 4) / 8;
        int16_t r = B_AT(0,0);
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 0, b, g, r);
    }

    // Upper right:
    {
        B_SET_CENTER(1,0);
        int16_t b = (10 * B_AT(0,0)
                   + 8 * (B_AT(0,-1) + B_AT(0,1))
                   - 2 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1) + B_AT(0,-2) + B_AT(0,2))
                   + B_AT(-2,0) + B_AT(2,0) + 8) / 16;
        int16_t g = B_AT(0,0);
        int16_t r = (10 * B_AT(0,0)
                   + 8 * (B_AT(-1,0) + B_AT(1,0))
                   - 2 * (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(-2,0) + B_AT(2,0))
                   + B_AT(0,-2) + B_AT(0,2) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 0, b, g, r);
    }

    // Lower left:
    {
        B_SET_CENTER(0,1);
        int16_t b = (10 * B_AT(0,0)
                   + 8 * (B_AT(-1,0) + B_AT(1,0))
                   - 2 * (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(-2,0) + B_AT(2,0))
                   + B_AT(0,-2) + B_AT(0,2) + 8) / 16;
        int16_t g = B_AT(0,0);
        int16_t r = (10 * B_AT(0,0)
                   + 8 * (B_AT(0,-1) + B_AT(0,1))
                   - 2 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1) + B_AT(0,-2) + B_AT(0,2))
                   + B_AT(-2,0) + B_AT(2,0) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 1, b, g, r);
    }

    // Lower right:
    {
        B_SET_CENTER(1,1);
        int16_t b = B_AT(0,0);
        int16_t g = (4 * B_AT(0,0)
                   + 2 * (B_AT(0,-1) + B_AT(0,1) + B_AT(-1,0) + B_AT(1,0))
                   - (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 4) / 8;
        int16_t r = (12 * B_AT(0,0)
                   + 4 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1))
                   - 3 * (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 1, b, g, r);
    }
}

__global__ void bggr_malvar2004(
    const uint8_t* raw,
    size_t raw_pitch,
    uint8_t* bgr,
    size_t bgr_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // We are looking at a 2x2 block
    const uint8_t* block = reinterpret_cast<const uint8_t*>(raw + (y * raw_pitch + x) * 2);
    uint8_t* bgr_block = reinterpret_cast<uint8_t*>(bgr + (y * bgr_pitch + x * 3) * 2);

    /*
        BGGR layout:

            B G
            G R

        Just swap the R and B channels from RGGB code.
    */

    // Upper left:
    {
        B_SET_CENTER(0,0);
        int16_t b = B_AT(0,0);
        int16_t g = (4 * B_AT(0,0)
                   + 2 * (B_AT(0,-1) + B_AT(0,1) + B_AT(-1,0) + B_AT(1,0))
                   - (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 4) / 8;
        int16_t r = (12 * B_AT(0,0)
                   + 4 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1))
                   - 3 * (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 0, b, g, r);
    }

    // Upper right:
    {
        B_SET_CENTER(1,0);
        int16_t b = (10 * B_AT(0,0)
                   + 8 * (B_AT(-1,0) + B_AT(1,0))
                   - 2 * (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(-2,0) + B_AT(2,0))
                   + B_AT(0,-2) + B_AT(0,2) + 8) / 16;
        int16_t g = B_AT(0,0);
        int16_t r = (10 * B_AT(0,0)
                   + 8 * (B_AT(0,-1) + B_AT(0,1))
                   - 2 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1) + B_AT(0,-2) + B_AT(0,2))
                   + B_AT(-2,0) + B_AT(2,0) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 0, b, g, r);
    }

    // Lower left:
    {
        B_SET_CENTER(0,1);
        int16_t b = (10 * B_AT(0,0)
                   + 8 * (B_AT(0,-1) + B_AT(0,1))
                   - 2 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1) + B_AT(0,-2) + B_AT(0,2))
                   + B_AT(-2,0) + B_AT(2,0) + 8) / 16;
        int16_t g = B_AT(0,0);
        int16_t r = (10 * B_AT(0,0)
                   + 8 * (B_AT(-1,0) + B_AT(1,0))
                   - 2 * (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(-2,0) + B_AT(2,0))
                   + B_AT(0,-2) + B_AT(0,2) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 1, b, g, r);
    }

    // Lower right:
    {
        B_SET_CENTER(1,1);
        int16_t b = (12 * B_AT(0,0)
                   + 4 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1))
                   - 3 * (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 8) / 16;
        int16_t g = (4 * B_AT(0,0)
                   + 2 * (B_AT(0,-1) + B_AT(0,1) + B_AT(-1,0) + B_AT(1,0))
                   - (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 4) / 8;
        int16_t r = B_AT(0,0);
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 1, b, g, r);
    }
}


//------------------------------------------------------------------------------
// Bilinear Algorithm

__global__ void rggb_bilinear(
    const uint8_t* raw,
    size_t raw_pitch,
    uint8_t* bgr,
    size_t bgr_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // We are looking at a 2x2 block
    const uint8_t* block = reinterpret_cast<const uint8_t*>(raw + (y * raw_pitch + x) * 2);
    uint8_t* bgr_block = reinterpret_cast<uint8_t*>(bgr + (y * bgr_pitch + x * 3) * 2);

    /*
        RGGB layout:

            R G
            G B

        G at R/B:

            0 G 0
            G x G
            0 G 0

        R at G, R column:

            G R G
            B x B
            G R G

        R at G, B column:

            G B G
            R x R
            G B G

        B at G, B column:

            G B G
            R x R
            G B G

        B at G, R column:

            G R G
            B x B
            G R G

        R at B, B column:

            R G R
            G x G
            R G R

        B at R, R column:

            B G B
            G x G
            B G B
    */

    // Upper left:
    {
        B_SET_CENTER(0,0);
        int16_t b = (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + 2) / 4;
        int16_t g = (B_AT(1,0) + B_AT(-1,0) + B_AT(0,1) + B_AT(0,-1) + 2) / 4;
        int16_t r = B_AT(0,0);
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 0, b, g, r);
    }

    // Upper right:
    {
        B_SET_CENTER(1,0);
        int16_t b = (B_AT(0,1) + B_AT(0,-1) + 1) / 2;
        int16_t g = B_AT(0,0);
        int16_t r = (B_AT(1,0) + B_AT(-1,0) + 1) / 2;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 0, b, g, r);
    }

    // Lower left:
    {
        B_SET_CENTER(0,1);
        int16_t b = (B_AT(1,0) + B_AT(-1,0) + 1) / 2;
        int16_t g = B_AT(0,0);
        int16_t r = (B_AT(0,1) + B_AT(0,-1) + 1) / 2;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 1, b, g, r);
    }

    // Lower right:
    {
        B_SET_CENTER(1,1);
        int16_t b = B_AT(0,0);
        int16_t g = (B_AT(1,0) + B_AT(-1,0) + B_AT(0,1) + B_AT(0,-1) + 2) / 4;
        int16_t r = (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + 2) / 4;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 1, b, g, r);
    }
}


//------------------------------------------------------------------------------
// Menon 2007 Algorithm

// This is an implementation of [2] with some modifications:
// (1) Rounding is applied to all averages.
// (2) When V/H gradients are close, the average is taken instead of picking.
// The second change boosts PSNR from 36.9494 + 0.7413 dB = 37.6907 dB.

#define ENABLE_CLOSE_AVERAGING

__global__ void rggb_menon2007_g(
    const uint8_t* raw,
    int raw_pitch,
    uint8_t* bgr,
    int bgr_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // We are looking at a 2x2 block
    const uint8_t* block = reinterpret_cast<const uint8_t*>(raw + (y * raw_pitch + x) * 2);
    uint8_t* bgr_block = reinterpret_cast<uint8_t*>(bgr + (y * bgr_pitch + x * 3) * 2);

    /*
        R G
        G B
    */

    // Upper Left:
    {
        const uint8_t* P = block;
        uint8_t* bgr = bgr_block;

        // Estimate G at R pixel
        int16_t G_h = 0;
        int16_t G_v = 0;

        // Horizontal estimation
        int16_t G_left = P[-1];
        int16_t G_right = P[1];
        int16_t R_center = P[0];
        int16_t R_left = P[-2];
        int16_t R_right = P[2];

        G_h = ((G_left + G_right + 1) >> 1) + ((2 * R_center - R_left - R_right + 2) >> 2);

        // Vertical estimation
        int16_t G_up = P[-raw_pitch];
        int16_t G_down = P[raw_pitch];
        int16_t R_up = P[-2 * raw_pitch];
        int16_t R_down = P[2 * raw_pitch];

        G_v = ((G_up + G_down + 1) >> 1) + ((2 * R_center - R_up - R_down + 2) >> 2);

        // Compute classifiers S_h and S_v
        int16_t C_center_h = R_center - G_h;
        int16_t C_left = R_left - G_left;
        int16_t C_right = R_right - G_right;
        int16_t S_h = abs(C_center_h - C_left) + abs(C_center_h - C_right);

        int16_t C_center_v = R_center - G_v;
        int16_t C_up = R_up - G_up;
        int16_t C_down = R_down - G_down;
        int16_t S_v = abs(C_center_v - C_up) + abs(C_center_v - C_down);

        // Decision
        int16_t G_est = (S_h <= S_v) ? G_h : G_v;
#ifdef ENABLE_CLOSE_AVERAGING
        if (abs(S_h - S_v) <= 27) {
            G_est = (G_h + G_v + 1) >> 1;
        }
#endif

        //bgr[0] = 255;
        bgr[1] = saturate_cast_int16_to_uint8(G_est);
        bgr[2] = P[0];
    }

    // Lower Right:
    {
        const uint8_t* P = block + raw_pitch + 1;
        uint8_t* bgr = bgr_block + bgr_pitch + 3;

        // Estimate G at B pixel
        int16_t G_h = 0;
        int16_t G_v = 0;

        // Horizontal estimation
        int16_t G_left = P[-1];
        int16_t G_right = P[1];
        int16_t B_center = P[0];
        int16_t B_left = P[-2];
        int16_t B_right = P[2];

        G_h = ((G_left + G_right + 1) >> 1) + ((2 * B_center - B_left - B_right + 2) >> 2);

        // Vertical estimation
        int16_t G_up = P[-raw_pitch];
        int16_t G_down = P[raw_pitch];
        int16_t B_up = P[-2 * raw_pitch];
        int16_t B_down = P[2 * raw_pitch];

        G_v = ((G_up + G_down + 1) >> 1) + ((2 * B_center - B_up - B_down + 2) >> 2);

        // Compute classifiers S_h and S_v
        int16_t C_center_h = B_center - G_h;
        int16_t C_left = B_left - G_left;
        int16_t C_right = B_right - G_right;
        int16_t S_h = abs(C_center_h - C_left) + abs(C_center_h - C_right);

        int16_t C_center_v = B_center - G_v;
        int16_t C_up = B_up - G_up;
        int16_t C_down = B_down - G_down;
        int16_t S_v = abs(C_center_v - C_up) + abs(C_center_v - C_down);

        // Decision
        int16_t G_est = (S_h <= S_v) ? G_h : G_v;
#ifdef ENABLE_CLOSE_AVERAGING
        if (abs(S_h - S_v) <= 29) {
            G_est = (G_h + G_v + 1) >> 1;
        }
#endif

        // Clamp the value
        bgr[0] = P[0];
        bgr[1] = saturate_cast_int16_to_uint8(G_est);
        //bgr[2] = 255;
    }

    // Upper Right:
    {
        const uint8_t* P = block + 1;
        uint8_t* bgr = bgr_block + 3;

        //bgr[0] = 255;
        bgr[1] = P[0];
        //bgr[2] = 255;
    }

    // Lower Left:
    {
        const uint8_t* P = block + raw_pitch;
        uint8_t* bgr = bgr_block + bgr_pitch;

        //bgr[0] = 255;
        bgr[1] = P[0];
        //bgr[2] = 255;
    }
}

__global__ void rggb_menon2007_rb(
    uint8_t* bgr,
    int bgr_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // We are looking at a 2x2 block
    uint8_t* bgr_block = reinterpret_cast<uint8_t*>(bgr + (y * bgr_pitch + x * 3) * 2);

    /*
        R G
        G B
    */

    // Upper Left:
    {
        uint8_t* bgr_pixel = bgr_block; // Pointer to the pixel at P0

        // Coordinates of the current pixel
        // For simplicity, we assume that the image is padded sufficiently

        // Edge-Directed Interpolation of (B - G) at red pixel
        // Using diagonal neighbors for interpolation

        // Get neighboring blue and green values
        int16_t B_UR = bgr_pixel[0 + 3 - bgr_pitch];       // Blue at (i - 1, j + 1)
        int16_t B_LL = bgr_pixel[0 - 3 + bgr_pitch];       // Blue at (i + 1, j - 1)
        int16_t G_UR = bgr_pixel[1 + 3 - bgr_pitch];       // Green at (i - 1, j + 1)
        int16_t G_LL = bgr_pixel[1 - 3 + bgr_pitch];       // Green at (i + 1, j - 1)

        int16_t B_UL = bgr_pixel[0 - 3 - bgr_pitch];       // Blue at (i - 1, j - 1)
        int16_t B_LR = bgr_pixel[0 + 3 + bgr_pitch];       // Blue at (i + 1, j + 1)
        int16_t G_UL = bgr_pixel[1 - 3 - bgr_pitch];       // Green at (i - 1, j - 1)
        int16_t G_LR = bgr_pixel[1 + 3 + bgr_pitch];       // Green at (i + 1, j + 1)

        // Compute color differences
        int16_t CD_UR = B_UR - G_UR;
        int16_t CD_LL = B_LL - G_LL;
        int16_t CD_UL = B_UL - G_UL;
        int16_t CD_LR = B_LR - G_LR;

        // Horizontal and Vertical estimates of (B - G)
        int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
        int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

        // Compute gradients for edge detection
        int16_t Grad_h = abs((CD_UL - CD_LR));
        int16_t Grad_v = abs((CD_UR - CD_LL));

        // Decision based on gradients
        int16_t CD_est = (Grad_h <= Grad_v) ? CD_h : CD_v;
#ifdef ENABLE_CLOSE_AVERAGING
        if (abs(Grad_h - Grad_v) <= 26) {
            CD_est = (CD_h + CD_v + 1) >> 1;
        }
#endif

        // Estimate Blue value at red pixel
        int16_t G_center = bgr_pixel[1];   // Green at current red pixel
        int16_t B_est = G_center + CD_est;

        // Clamp the value
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }

    // Lower Right
    {
        uint8_t* bgr_pixel = bgr_block + bgr_pitch + 3; // Pointer to the pixel at P3

        // Get neighboring red and green values
        int16_t R_UL = bgr_pixel[2 - 3 - bgr_pitch];    // Red at (i - 1, j - 1)
        int16_t R_LR = bgr_pixel[2 + 3 + bgr_pitch];    // Red at (i + 1, j + 1)
        int16_t G_UL = bgr_pixel[1 - 3 - bgr_pitch];    // Green at (i - 1, j - 1)
        int16_t G_LR = bgr_pixel[1 + 3 + bgr_pitch];    // Green at (i + 1, j + 1)

        int16_t R_UR = bgr_pixel[2 + 3 - bgr_pitch];    // Red at (i - 1, j + 1)
        int16_t R_LL = bgr_pixel[2 - 3 + bgr_pitch];    // Red at (i + 1, j - 1)
        int16_t G_UR = bgr_pixel[1 + 3 - bgr_pitch];    // Green at (i - 1, j + 1)
        int16_t G_LL = bgr_pixel[1 - 3 + bgr_pitch];    // Green at (i + 1, j - 1)

        // Compute color differences
        int16_t CD_UL = R_UL - G_UL;
        int16_t CD_LR = R_LR - G_LR;
        int16_t CD_UR = R_UR - G_UR;
        int16_t CD_LL = R_LL - G_LL;

        // Horizontal and Vertical estimates of (R - G)
        int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
        int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

        // Compute gradients for edge detection
        int16_t Grad_h = abs((CD_UL - CD_LR));
        int16_t Grad_v = abs((CD_UR - CD_LL));

        // Decision based on gradients
        int16_t CD_est = (Grad_h <= Grad_v) ? CD_h : CD_v;
#ifdef ENABLE_CLOSE_AVERAGING
        if (abs(Grad_h - Grad_v) <= 26) {
            CD_est = (CD_h + CD_v + 1) >> 1;
        }
#endif

        // Estimate Red value at blue pixel
        int16_t G_center = bgr_pixel[1];   // Green at current blue pixel
        int16_t R_est = G_center + CD_est;

        // Clamp the value
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
    }

    // Lower Left
    {
        uint8_t* bgr_pixel = bgr_block + bgr_pitch; // Pointer to the pixel at P2

        // Estimate Red at green pixel using bilinear interpolation of (R - G)
        int16_t R_up = bgr_pixel[2 - bgr_pitch];       // Red at (i - 1, j)
        int16_t R_down = bgr_pixel[2 + bgr_pitch];     // Red at (i + 1, j)
        int16_t G_up = bgr_pixel[1 - bgr_pitch];
        int16_t G_down = bgr_pixel[1 + bgr_pitch];

        int16_t CD_RU = R_up - G_up;
        int16_t CD_RD = R_down - G_down;

        int16_t CD_R = (CD_RU + CD_RD + 1) >> 1;
        int16_t G_center = bgr_pixel[1];
        int16_t R_est = G_center + CD_R;

        // Estimate Blue at green pixel using bilinear interpolation of (B - G)
        int16_t B_left = bgr_pixel[0 - 3];         // Blue at (i, j - 1)
        int16_t B_right = bgr_pixel[0 + 3];        // Blue at (i, j + 1)
        int16_t G_left = bgr_pixel[1 - 3];
        int16_t G_right = bgr_pixel[1 + 3];

        int16_t CD_BL = B_left - G_left;
        int16_t CD_BR = B_right - G_right;

        int16_t CD_B = (CD_BL + CD_BR + 1) >> 1;
        int16_t B_est = G_center + CD_B;

        // Clamp the values
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }

    // Upper Right
    {
        uint8_t* bgr_pixel = bgr_block + 3; // Pointer to the pixel at P1

        // Estimate Red at green pixel using bilinear interpolation of (R - G)
        int16_t R_left = bgr_pixel[2 - 3];       // Red at (i, j - 1)
        int16_t R_right = bgr_pixel[2 + 3];      // Red at (i, j + 1)
        int16_t G_left = bgr_pixel[1 - 3];       // Green at (i, j - 1)
        int16_t G_right = bgr_pixel[1 + 3];      // Green at (i, j + 1)

        int16_t CD_RL = R_left - G_left;
        int16_t CD_RR = R_right - G_right;

        int16_t CD_R = (CD_RL + CD_RR + 1) >> 1;
        int16_t G_center = bgr_pixel[1];
        int16_t R_est = G_center + CD_R;

        // Estimate Blue at green pixel using bilinear interpolation of (B - G)
        int16_t B_up = bgr_pixel[0 - bgr_pitch]; // Blue at (i - 1, j)
        int16_t B_down = bgr_pixel[0 + bgr_pitch]; // Blue at (i + 1, j)
        int16_t G_up = bgr_pixel[1 - bgr_pitch];
        int16_t G_down = bgr_pixel[1 + bgr_pitch];

        int16_t CD_BU = B_up - G_up;
        int16_t CD_BD = B_down - G_down;

        int16_t CD_B = (CD_BU + CD_BD + 1) >> 1;
        int16_t B_est = G_center + CD_B;

        // Clamp the values
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }
}

__global__ void bggr_menon2007_g(
    const uint8_t* raw,
    int raw_pitch,
    uint8_t* bgr,
    int bgr_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index in 2x2 blocks
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index in 2x2 blocks

    // Calculate the starting index of the 2x2 block
    const uint8_t* block = raw + (y * raw_pitch + x) * 2;
    uint8_t* bgr_block = bgr + (y * bgr_pitch + x * 3) * 2;

    /*
        BGGR pattern in a 2x2 block:

            B G
            G R
    */

    // Upper Left (P0): B pixel
    {
        const uint8_t* P = block;
        uint8_t* bgr = bgr_block;

        // Estimate G at B pixel
        int16_t G_h = 0;
        int16_t G_v = 0;

        // Horizontal estimation
        int16_t G_left = P[-1];        // G at (i, j - 1)
        int16_t G_right = P[1];        // G at (i, j + 1)
        int16_t B_center = P[0];       // B at (i, j)
        int16_t B_left = P[-2];        // B at (i, j - 2)
        int16_t B_right = P[2];        // B at (i, j + 2)

        G_h = ((G_left + G_right + 1) >> 1) + ((2 * B_center - B_left - B_right + 2) >> 2);

        // Vertical estimation
        int16_t G_up = P[-raw_pitch];        // G at (i - 1, j)
        int16_t G_down = P[raw_pitch];       // G at (i + 1, j)
        int16_t B_up = P[-2 * raw_pitch];    // B at (i - 2, j)
        int16_t B_down = P[2 * raw_pitch];   // B at (i + 2, j)

        G_v = ((G_up + G_down + 1) >> 1) + ((2 * B_center - B_up - B_down + 2) >> 2);

        // Compute classifiers S_h and S_v
        int16_t C_center_h = B_center - G_h;
        int16_t C_left = B_left - G_left;
        int16_t C_right = B_right - G_right;
        int16_t S_h = abs(C_center_h - C_left) + abs(C_center_h - C_right);

        int16_t C_center_v = B_center - G_v;
        int16_t C_up = B_up - G_up;
        int16_t C_down = B_down - G_down;
        int16_t S_v = abs(C_center_v - C_up) + abs(C_center_v - C_down);

        // Decision
        int16_t G_est = (S_h <= S_v) ? G_h : G_v;
#ifdef ENABLE_CLOSE_AVERAGING
        if (abs(S_h - S_v) <= 29) {
            G_est = (G_h + G_v + 1) >> 1;
        }
#endif

        // Clamp the value
        bgr[0] = P[0]; // B value
        bgr[1] = saturate_cast_int16_to_uint8(G_est);
        // bgr[2] = 255; // R value (to be filled in later)
    }

    // Upper Right (P1): G pixel
    {
        const uint8_t* P = block + 1;
        uint8_t* bgr = bgr_block + 3;

        // G pixel, green value is known
        bgr[1] = P[0];
    }

    // Lower Left (P2): G pixel
    {
        const uint8_t* P = block + raw_pitch;
        uint8_t* bgr = bgr_block + bgr_pitch;

        // G pixel, green value is known
        bgr[1] = P[0];
    }

    // Lower Right (P3): R pixel
    {
        const uint8_t* P = block + raw_pitch + 1;
        uint8_t* bgr = bgr_block + bgr_pitch + 3;

        // Estimate G at R pixel
        int16_t G_h = 0;
        int16_t G_v = 0;

        // Horizontal estimation
        int16_t G_left = P[-1];        // G at (i, j - 1)
        int16_t G_right = P[1];        // G at (i, j + 1)
        int16_t R_center = P[0];       // R at (i, j)
        int16_t R_left = P[-2];        // R at (i, j - 2)
        int16_t R_right = P[2];        // R at (i, j + 2)

        G_h = ((G_left + G_right + 1) >> 1) + ((2 * R_center - R_left - R_right + 2) >> 2);

        // Vertical estimation
        int16_t G_up = P[-raw_pitch];        // G at (i - 1, j)
        int16_t G_down = P[raw_pitch];       // G at (i + 1, j)
        int16_t R_up = P[-2 * raw_pitch];    // R at (i - 2, j)
        int16_t R_down = P[2 * raw_pitch];   // R at (i + 2, j)

        G_v = ((G_up + G_down + 1) >> 1) + ((2 * R_center - R_up - R_down + 2) >> 2);

        // Compute classifiers S_h and S_v
        int16_t C_center_h = R_center - G_h;
        int16_t C_left = R_left - G_left;
        int16_t C_right = R_right - G_right;
        int16_t S_h = abs(C_center_h - C_left) + abs(C_center_h - C_right);

        int16_t C_center_v = R_center - G_v;
        int16_t C_up = R_up - G_up;
        int16_t C_down = R_down - G_down;
        int16_t S_v = abs(C_center_v - C_up) + abs(C_center_v - C_down);

        // Decision
        int16_t G_est = (S_h <= S_v) ? G_h : G_v;
#ifdef ENABLE_CLOSE_AVERAGING
        if (abs(S_h - S_v) <= 27) {
            G_est = (G_h + G_v + 1) >> 1;
        }
#endif

        // Clamp the value
        // bgr[0] = 255; // B value (to be filled in later)
        bgr[1] = saturate_cast_int16_to_uint8(G_est);
        bgr[2] = P[0]; // R value
    }
}

__global__ void bggr_menon2007_rb(
    uint8_t* bgr,
    int bgr_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index in 2x2 blocks
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index in 2x2 blocks

    // Calculate the starting index of the 2x2 block
    uint8_t* bgr_block = bgr + (y * bgr_pitch + x * 3) * 2;

    /*
        BGGR pattern in a 2x2 block:

        Positions:
        P0: Upper Left (B pixel)
        P1: Upper Right (G pixel)
        P2: Lower Left (G pixel)
        P3: Lower Right (R pixel)
    */

    // Upper Left (P0): B pixel
    {
        uint8_t* bgr_pixel = bgr_block; // Pointer to the pixel at P0

        // Estimate Red at Blue pixel (similar to estimating Blue at Red pixel in RGGB)
        // Get neighboring red and green values
        int16_t R_UR = bgr_pixel[2 + 3 - bgr_pitch];    // Red at (i - 1, j + 1)
        int16_t R_LL = bgr_pixel[2 - 3 + bgr_pitch];    // Red at (i + 1, j - 1)
        int16_t G_UR = bgr_pixel[1 + 3 - bgr_pitch];    // Green at (i - 1, j + 1)
        int16_t G_LL = bgr_pixel[1 - 3 + bgr_pitch];    // Green at (i + 1, j - 1)

        int16_t R_UL = bgr_pixel[2 - 3 - bgr_pitch];    // Red at (i - 1, j - 1)
        int16_t R_LR = bgr_pixel[2 + 3 + bgr_pitch];    // Red at (i + 1, j + 1)
        int16_t G_UL = bgr_pixel[1 - 3 - bgr_pitch];    // Green at (i - 1, j - 1)
        int16_t G_LR = bgr_pixel[1 + 3 + bgr_pitch];    // Green at (i + 1, j + 1)

        // Compute color differences
        int16_t CD_UR = R_UR - G_UR;
        int16_t CD_LL = R_LL - G_LL;
        int16_t CD_UL = R_UL - G_UL;
        int16_t CD_LR = R_LR - G_LR;

        // Horizontal and Vertical estimates of (R - G)
        int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
        int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

        // Compute gradients for edge detection
        int16_t Grad_h = abs(CD_UL - CD_LR);
        int16_t Grad_v = abs(CD_UR - CD_LL);

        // Decision based on gradients
        int16_t CD_est = (Grad_h <= Grad_v) ? CD_h : CD_v;
#ifdef ENABLE_CLOSE_AVERAGING
        if (abs(Grad_h - Grad_v) <= 26) {
            CD_est = (CD_h + CD_v + 1) >> 1;
        }
#endif

        // Estimate Red value at blue pixel
        int16_t G_center = bgr_pixel[1];   // Green at current blue pixel
        int16_t R_est = G_center + CD_est;

        // Clamp the value
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
    }

    // Lower Right (P3): R pixel
    {
        uint8_t* bgr_pixel = bgr_block + bgr_pitch + 3; // Pointer to the pixel at P3

        // Estimate Blue at Red pixel (similar to estimating Red at Blue pixel in RGGB)
        // Get neighboring blue and green values
        int16_t B_UR = bgr_pixel[0 + 3 - bgr_pitch];    // Blue at (i - 1, j + 1)
        int16_t B_LL = bgr_pixel[0 - 3 + bgr_pitch];    // Blue at (i + 1, j - 1)
        int16_t G_UR = bgr_pixel[1 + 3 - bgr_pitch];    // Green at (i - 1, j + 1)
        int16_t G_LL = bgr_pixel[1 - 3 + bgr_pitch];    // Green at (i + 1, j - 1)

        int16_t B_UL = bgr_pixel[0 - 3 - bgr_pitch];    // Blue at (i - 1, j - 1)
        int16_t B_LR = bgr_pixel[0 + 3 + bgr_pitch];    // Blue at (i + 1, j + 1)
        int16_t G_UL = bgr_pixel[1 - 3 - bgr_pitch];    // Green at (i - 1, j - 1)
        int16_t G_LR = bgr_pixel[1 + 3 + bgr_pitch];    // Green at (i + 1, j + 1)

        // Compute color differences
        int16_t CD_UR = B_UR - G_UR;
        int16_t CD_LL = B_LL - G_LL;
        int16_t CD_UL = B_UL - G_UL;
        int16_t CD_LR = B_LR - G_LR;

        // Horizontal and Vertical estimates of (B - G)
        int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
        int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

        // Compute gradients for edge detection
        int16_t Grad_h = abs(CD_UL - CD_LR);
        int16_t Grad_v = abs(CD_UR - CD_LL);

        // Decision based on gradients
        int16_t CD_est = (Grad_h <= Grad_v) ? CD_h : CD_v;
#ifdef ENABLE_CLOSE_AVERAGING
        if (abs(Grad_h - Grad_v) <= 26) {
            CD_est = (CD_h + CD_v + 1) >> 1;
        }
#endif

        // Estimate Blue value at red pixel
        int16_t G_center = bgr_pixel[1];   // Green at current red pixel
        int16_t B_est = G_center + CD_est;

        // Clamp the value
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }

    // Upper Right (P1): G pixel
    {
        uint8_t* bgr_pixel = bgr_block + 3; // Pointer to the pixel at P1

        // Estimate Red at green pixel using bilinear interpolation of (R - G)
        int16_t R_up = bgr_pixel[2 - bgr_pitch];       // Red at (i - 1, j)
        int16_t R_down = bgr_pixel[2 + bgr_pitch];     // Red at (i + 1, j)
        int16_t G_up = bgr_pixel[1 - bgr_pitch];
        int16_t G_down = bgr_pixel[1 + bgr_pitch];

        int16_t CD_RU = R_up - G_up;
        int16_t CD_RD = R_down - G_down;

        int16_t CD_R = (CD_RU + CD_RD + 1) >> 1;
        int16_t G_center = bgr_pixel[1];
        int16_t R_est = G_center + CD_R;

        // Estimate Blue at green pixel using bilinear interpolation of (B - G)
        int16_t B_left = bgr_pixel[0 - 3];         // Blue at (i, j - 1)
        int16_t B_right = bgr_pixel[0 + 3];        // Blue at (i, j + 1)
        int16_t G_left = bgr_pixel[1 - 3];
        int16_t G_right = bgr_pixel[1 + 3];

        int16_t CD_BL = B_left - G_left;
        int16_t CD_BR = B_right - G_right;

        int16_t CD_B = (CD_BL + CD_BR + 1) >> 1;
        int16_t B_est = G_center + CD_B;

        // Clamp the values
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }

    // Lower Left (P2): G pixel
    {
        uint8_t* bgr_pixel = bgr_block + bgr_pitch; // Pointer to the pixel at P2

        // Estimate Red at green pixel using bilinear interpolation of (R - G)
        int16_t R_left = bgr_pixel[2 - 3];       // Red at (i, j - 1)
        int16_t R_right = bgr_pixel[2 + 3];      // Red at (i, j + 1)
        int16_t G_left = bgr_pixel[1 - 3];       // Green at (i, j - 1)
        int16_t G_right = bgr_pixel[1 + 3];      // Green at (i, j + 1)

        int16_t CD_RL = R_left - G_left;
        int16_t CD_RR = R_right - G_right;

        int16_t CD_R = (CD_RL + CD_RR + 1) >> 1;
        int16_t G_center = bgr_pixel[1];
        int16_t R_est = G_center + CD_R;

        // Estimate Blue at green pixel using bilinear interpolation of (B - G)
        int16_t B_up = bgr_pixel[0 - bgr_pitch]; // Blue at (i - 1, j)
        int16_t B_down = bgr_pixel[0 + bgr_pitch]; // Blue at (i + 1, j)
        int16_t G_up = bgr_pixel[1 - bgr_pitch];
        int16_t G_down = bgr_pixel[1 + bgr_pitch];

        int16_t CD_BU = B_up - G_up;
        int16_t CD_BD = B_down - G_down;

        int16_t CD_B = (CD_BU + CD_BD + 1) >> 1;
        int16_t B_est = G_center + CD_B;

        // Clamp the values
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }
}
