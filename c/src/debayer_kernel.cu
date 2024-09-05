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
// Saronic1 Algorithm

__global__ void rggb_saronic1_g(
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

        // Calculate G:
        int16_t g_left = P[-1];
        int16_t g_right = P[1];
        int16_t g_up = P[-raw_pitch];
        int16_t g_down = P[raw_pitch];

        int16_t rb_left = P[-2];
        int16_t rb_right = P[2];
        int16_t rb_center = P[0];
        int16_t rb_up = P[-2 * raw_pitch];
        int16_t rb_down = P[2 * raw_pitch];

        int16_t dg_y = abs(g_down - g_up);
        int16_t dg_x = abs(g_right - g_left);

        int16_t fy = (g_up + g_down) * 2 + (2 * rb_center - rb_up - rb_down);
        int16_t fx = (g_left + g_right) * 2 + (2 * rb_center - rb_left - rb_right);

#if 0
        // Edge-aware: Use more stable estimate
        const int16_t T = 34;
        fy = (dg_y >= dg_x + T) ? fx : fy;
        fx = (dg_x >= dg_y + T) ? fy : fx;
#endif

        //bgr[0] = 255;
        bgr[1] = saturate_cast_int16_to_uint8((fx + fy + 4) / 8);
        bgr[2] = P[0];
    }

    // Lower Right:
    {
        const uint8_t* P = block + raw_pitch + 1;
        uint8_t* bgr = bgr_block + bgr_pitch + 3;

        // Calculate G:
        int16_t g_left = P[-1];
        int16_t g_right = P[1];
        int16_t g_up = P[-raw_pitch];
        int16_t g_down = P[raw_pitch];

        int16_t rb_left = P[-2];
        int16_t rb_right = P[2];
        int16_t rb_center = P[0];
        int16_t rb_up = P[-2 * raw_pitch];
        int16_t rb_down = P[2 * raw_pitch];

        int16_t dg_y = abs(g_down - g_up);
        int16_t dg_x = abs(g_right - g_left);

        int16_t fy = (g_up + g_down) * 2 + (2 * rb_center - rb_up - rb_down);
        int16_t fx = (g_left + g_right) * 2 + (2 * rb_center - rb_left - rb_right);

#if 0
        // Edge-aware: Use more stable estimate
        const int16_t T = 34;
        fy = (dg_y >= dg_x + T) ? fx : fy;
        fx = (dg_x >= dg_y + T) ? fy : fx;
#endif

        bgr[0] = P[0];
        bgr[1] = saturate_cast_int16_to_uint8((fx + fy + 4) / 8);
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

__global__ void rggb_saronic1_rb(
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
        uint8_t* bgr = bgr_block;

        // Calculate B at R:
        /*
            B G B
            G x G
            B G B
        */
        int16_t b_ur = bgr[3-bgr_pitch];
        int16_t b_ul = bgr[-3-bgr_pitch];
        int16_t b_ll = bgr[-3+bgr_pitch];
        int16_t b_lr = bgr[3+bgr_pitch];

#if 0
        int16_t g_center = bgr[1];
        int16_t g_ur = bgr[1+3-bgr_pitch];
        int16_t g_ul = bgr[1-3-bgr_pitch];
        int16_t g_ll = bgr[1-3+bgr_pitch];
        int16_t g_lr = bgr[1+3+bgr_pitch];

        int16_t dg_x = abs(g_ur - g_ll);
        int16_t dg_y = abs(g_ul - g_lr);

        int16_t fx = (b_ur + b_ll) * 2 + 2 * g_center - g_ur - g_ll;
        int16_t fy = (b_ul + b_lr) * 2 + 2 * g_center - g_ul - g_lr;

        // Edge-aware: Use more stable estimate
        const int16_t T = 34;
        fy = (dg_y >= dg_x + T) ? fx : fy;
        fx = (dg_x >= dg_y + T) ? fy : fx;

        bgr[0] = saturate_cast_int16_to_uint8((fx + fy + 4) / 8);
#else
        int16_t r_up = bgr[2 - bgr_pitch*2];
        int16_t r_down = bgr[2 + bgr_pitch*2];
        int16_t r_left = bgr[2 - 6];
        int16_t r_right = bgr[2 + 6];
        int16_t r_center = bgr[2];

        int16_t fx = (r_center * 6 - (r_left + r_right) * 3) + (b_ur + b_ll) * 4;
        int16_t fy = (r_center * 6 - (r_down + r_up) * 3) + (b_lr + b_ul) * 4;

#if 0
        int16_t dg_x = abs(b_ur - b_ll);
        int16_t dg_y = abs(b_lr - b_ul);

        // Edge-aware: Use more stable estimate
        const int16_t T = 34;
        fy = (dg_y >= dg_x + T) ? fx : fy;
        fx = (dg_x >= dg_y + T) ? fy : fx;
#endif

        bgr[0] = saturate_cast_int16_to_uint8((fx + fy + 8) / 16);
#endif
    }

    // Lower Right
    {
        uint8_t* bgr = bgr_block + bgr_pitch + 3;

        // Calculate R at B:
        /*
            R G R
            G B G
            R G R
        */
        int16_t r_ur = bgr[2+3-bgr_pitch];
        int16_t r_ul = bgr[2-3-bgr_pitch];
        int16_t r_ll = bgr[2-3+bgr_pitch];
        int16_t r_lr = bgr[2+3+bgr_pitch];

#if 0
        int16_t g_center = bgr[1];
        int16_t g_ur = bgr[1+3-bgr_pitch];
        int16_t g_ul = bgr[1-3-bgr_pitch];
        int16_t g_ll = bgr[1-3+bgr_pitch];
        int16_t g_lr = bgr[1+3+bgr_pitch];

        int16_t dg_x = abs(g_ur - g_ll);
        int16_t dg_y = abs(g_ul - g_lr);

        int16_t fx = (r_ur + r_ll) * 2 + 2 * g_center - g_ur - g_ll;
        int16_t fy = (r_ul + r_lr) * 2 + 2 * g_center - g_ul - g_lr;

        // Edge-aware: Use more stable estimate
        const int16_t T = 34;
        fy = (dg_y >= dg_x + T) ? fx : fy;
        fx = (dg_x >= dg_y + T) ? fy : fx;

        bgr[2] = saturate_cast_int16_to_uint8((fx + fy + 4) / 8);
#else
        int16_t b_up = bgr[-bgr_pitch*2];
        int16_t b_down = bgr[bgr_pitch*2];
        int16_t b_left = bgr[-6];
        int16_t b_right = bgr[6];
        int16_t b_center = bgr[0];

        int16_t fx = (b_center * 6 - (b_left + b_right) * 3) + (r_ur + r_ll) * 4;
        int16_t fy = (b_center * 6 - (b_down + b_up) * 3) + (r_lr + r_ul) * 4;

#if 0
        int16_t dg_x = abs(r_ur - r_ll);
        int16_t dg_y = abs(r_lr - r_ul);

        // Edge-aware: Use more stable estimate
        const int16_t T = 34;
        fx = (dg_x >= dg_y + T) ? fy : fx;
        fy = (dg_y >= dg_x + T) ? fx : fy;
#endif

        bgr[2] = saturate_cast_int16_to_uint8((fx + fy + 8) / 16);
#endif
    }

    // Lower Left
    {
        uint8_t* bgr = bgr_block + bgr_pitch;

        int16_t g_center = bgr[1];
        int16_t g_left = bgr[1-3];
        int16_t g_right = bgr[1+3];
        int16_t g_up = bgr[1-bgr_pitch];
        int16_t g_down = bgr[1+bgr_pitch];

        int16_t b_left = bgr[-3];
        int16_t b_right = bgr[3];

        int16_t r_up = bgr[2 - bgr_pitch];
        int16_t r_down = bgr[2 + bgr_pitch];

        int16_t g_left2 = bgr[1-6];
        int16_t g_right2 = bgr[1+6];
        int16_t g_up2 = bgr[1-2*bgr_pitch];
        int16_t g_down2 = bgr[1+2*bgr_pitch];
        int16_t g_ul = bgr[1-3-bgr_pitch];
        int16_t g_ur = bgr[1+3-bgr_pitch];
        int16_t g_ll = bgr[1-3+bgr_pitch];
        int16_t g_lr = bgr[1+3+bgr_pitch];

#if 1
        int16_t x = 10 * g_center
            + 8 * (b_left + b_right)
            - 2 * (g_left + g_right + g_ul + g_ur + g_ll + g_lr)
            + g_up + g_down;
        bgr[0] = saturate_cast_int16_to_uint8((x + 8) / 16);
#else
        bgr[0] = saturate_cast_int16_to_uint8((2 * (b_left + b_right) + 2 * g_center - g_left - g_right + 2) / 4);
#endif

#if 1
        int16_t y = 10 * g_center
            + 8 * (r_up + r_down)
            - 2 * (g_up2 + g_down2 + g_ul + g_ur + g_ll + g_lr)
            + g_left2 + g_right2;
        bgr[2] = saturate_cast_int16_to_uint8((y + 8) / 16);
#else
        bgr[2] = saturate_cast_int16_to_uint8((2 * (r_up + r_down) + 2 * g_center - g_up - g_down + 2) / 4);
#endif
    }

    // Upper Right
    {
        uint8_t* bgr = bgr_block + 3;

        int16_t g_center = bgr[1];
        int16_t g_left = bgr[1-3];
        int16_t g_right = bgr[1+3];
        int16_t g_up = bgr[1-bgr_pitch];
        int16_t g_down = bgr[1+bgr_pitch];

        int16_t r_left = bgr[2 - 3];
        int16_t r_right = bgr[2 + 3];

        int16_t b_up = bgr[-bgr_pitch];
        int16_t b_down = bgr[bgr_pitch];

        int16_t g_left2 = bgr[1-6];
        int16_t g_right2 = bgr[1+6];
        int16_t g_up2 = bgr[1-2*bgr_pitch];
        int16_t g_down2 = bgr[1+2*bgr_pitch];
        int16_t g_ul = bgr[1-3-bgr_pitch];
        int16_t g_ur = bgr[1+3-bgr_pitch];
        int16_t g_ll = bgr[1-3+bgr_pitch];
        int16_t g_lr = bgr[1+3+bgr_pitch];

#if 1
        int16_t x = 10 * g_center
            + 8 * (b_up + b_down)
            - 2 * (g_up + g_down + g_ul + g_ur + g_ll + g_lr)
            + g_left + g_right;
        bgr[0] = saturate_cast_int16_to_uint8((x + 8) / 16);
#else
        bgr[0] = saturate_cast_int16_to_uint8((2 * (b_up + b_down) + 2 * g_center - g_up - g_down + 2) / 4);
#endif

#if 1
        int16_t y = 10 * g_center
            + 8 * (r_left + r_right)
            - 2 * (g_left2 + g_right2 + g_ul + g_ur + g_ll + g_lr)
            + g_up2 + g_down2;
        bgr[2] = saturate_cast_int16_to_uint8((y + 8) / 16);
#else
        bgr[2] = saturate_cast_int16_to_uint8((2 * (r_left + r_right) + 2 * g_center - g_left - g_right + 2) / 4);
#endif
    }
}
