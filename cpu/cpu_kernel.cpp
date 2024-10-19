/**
 * @file cpu_kernel.cpp
 * @brief CPU-based kernel functions for image padding and Bayer pattern demosaicing.
 */

#include "cpu_kernel.hpp"

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>

#define ENABLE_CLOSE_AVERAGING

/**
 * @brief Pads the top and bottom edges of an image by replicating the first and last rows.
 *
 * This function copies the first `width` pixels of the original image to the top padding
 * and the last `width` pixels to the bottom padding. The padding size is specified by `pad`.
 *
 * @param data Pointer to the image data buffer.
 * @param width Width of the original image in pixels.
 * @param height Height of the original image in pixels.
 * @param pitch Number of bytes per row in the image buffer (must be at least `width + 2 * pad`).
 * @param pad Number of padding rows to add to both the top and bottom.
 *
 * @pre `data` must not be `nullptr`.
 * @pre `width` and `height` must be positive.
 * @pre `pitch` must be at least `width + 2 * pad`.
 * @pre `pad` must be positive.
 *
 * @note The function assumes that the `data` buffer has been allocated with enough space
 *       to accommodate the additional padding rows.
 */
void padTopBottomEdges(uint8_t* data, int width, int height, int pitch, int pad) {
    // Validate input parameters
    assert(data != nullptr && "Data pointer must not be null.");
    assert(width > 0 && "Image width must be positive.");
    assert(height > 0 && "Image height must be positive.");
    assert(pitch >= (width + 2 * pad) && "Pitch must be at least width + 2 * pad.");
    assert(pad > 0 && "Padding size must be positive.");

    // Calculate pointers to the first and last rows of the original image
    uint8_t* original_first_row = data + pad * pitch + pad;
    uint8_t* original_last_row  = data + (pad + height - 1) * pitch + pad;

    // Pad the top edges by replicating the first row of the original image
    for (int y = 0; y < pad; ++y) {
        uint8_t* dest_top = data + y * pitch + pad;
        memcpy(dest_top, original_first_row, width * sizeof(uint8_t));
    }

    // Pad the bottom edges by replicating the last row of the original image
    for (int y = 0; y < pad; ++y) {
        uint8_t* dest_bottom = data + (pad + height + y) * pitch + pad;
        memcpy(dest_bottom, original_last_row, width * sizeof(uint8_t));
    }
}

/**
 * @brief Pads the left and right edges of an image by replicating the first and last columns.
 *
 * This function fills the left padding with the first column of the original image and the
 * right padding with the last column. The padding size is specified by `pad`.
 *
 * @param data Pointer to the image data buffer.
 * @param width Width of the original image in pixels.
 * @param height Height of the original image in pixels.
 * @param pitch Number of bytes per row in the image buffer (must be at least `width + 2 * pad`).
 * @param pad Number of padding columns to add to both the left and right.
 *
 * @pre `data` must not be `nullptr`.
 * @pre `width` and `height` must be positive.
 * @pre `pitch` must be at least `width + 2 * pad`.
 * @pre `pad` must be positive.
 *
 * @note The function assumes that the `data` buffer has been allocated with enough space
 *       to accommodate the additional padding columns.
 */
void padLeftRightEdges(uint8_t* data, int width, int height, int pitch, int pad) {
    // Validate input parameters
    assert(data != nullptr && "Data pointer must not be null.");
    assert(width > 0 && "Image width must be positive.");
    assert(height > 0 && "Image height must be positive.");
    assert(pitch >= (width + 2 * pad) && "Pitch must be at least width + 2 * pad.");
    assert(pad > 0 && "Padding size must be positive.");

    for (int y = 0; y < height; ++y) {
        uint8_t* original_row = data + (pad + y) * pitch + pad;
        uint8_t* dest_left  = data + (pad + y) * pitch;
        uint8_t* dest_right = data + (pad + y) * pitch + pad + width;

        uint8_t first_pixel = original_row[0];
        uint8_t last_pixel  = original_row[width - 1];

        // Use memset-like operations for padding
        std::memset(dest_left, first_pixel, pad);
        std::memset(dest_right, last_pixel, pad);
    }
}

/**
 * @brief Clamps a 16-bit signed integer to an 8-bit unsigned integer range [0, 255].
 *
 * This inline helper function ensures that the input value does not exceed the
 * bounds of an 8-bit unsigned integer.
 *
 * @param x The 16-bit signed integer to clamp.
 * @return The clamped 8-bit unsigned integer.
 */
inline uint8_t saturate_cast_int16_to_uint8(int16_t x) {
    if (x < 0) return 0;
    if (x > 255) return 255;
    return static_cast<uint8_t>(x);
}

/**
 * @brief Estimates and fills the Green channel in a BGGR Bayer pattern image.
 *
 * This function processes the raw Bayer data to estimate the Green channel values
 * using the Menon 2007 algorithm. It iterates over 2x2 pixel blocks and computes
 * the Green values based on horizontal and vertical gradients.
 *
 * @param raw Pointer to the raw Bayer data buffer.
 * @param raw_pitch Number of bytes per row in the raw Bayer data.
 * @param bgr Pointer to the output BGR data buffer (only B and G channels are filled).
 * @param bgr_pitch Number of bytes per row in the BGR data buffer.
 * @param width Width of the image in pixels (must be even).
 * @param height Height of the image in pixels (must be even).
 *
 * @pre `raw` and `bgr` must not be `nullptr`.
 * @pre `width` and `height` must be positive and even.
 * @pre `raw_pitch` and `bgr_pitch` must be sufficient to hold the respective image data.
 *
 * @note The Red channel will be filled in a separate function.
 */
void bggr_menon2007_g_cpu(
    const uint8_t* raw,
    int raw_pitch,
    uint8_t* bgr,
    int bgr_pitch,
    int width,
    int height)
{
    // Validate input parameters
    assert(raw  != nullptr && "Raw data pointer must not be null.");
    assert(bgr   != nullptr && "BGR data pointer must not be null.");
    assert(width > 0 && "Image width must be positive.");
    assert(height > 0 && "Image height must be positive.");
    assert(width % 2 == 0 && "Image width must be even.");
    assert(height % 2 == 0 && "Image height must be even.");
    assert(raw_pitch >= width && "Raw pitch must be at least equal to image width.");
    assert(bgr_pitch >= width * 3 && "BGR pitch must be at least equal to image width * 3.");

    // Iterate over the image in 2x2 blocks
    for(int y = 0; y < height; y += 2) {
        for(int x = 0; x < width; x += 2) {
            // Calculate the starting index of the 2x2 block in raw and BGR buffers
            const uint8_t* block = raw + y * raw_pitch + x;
            uint8_t* bgr_block   = bgr + y * bgr_pitch + x * 3;

            // --- Upper Left Pixel (P0): Blue Pixel ---
            {
                const uint8_t* P = block;
                uint8_t* bgr_p0    = bgr_block;

                // Initialize estimations for Green channel
                int16_t G_h = 0; // Horizontal estimation
                int16_t G_v = 0; // Vertical estimation

                // Horizontal estimation
                // Ensure that we do not access out-of-bounds memory
                // This assumes that the image has been appropriately padded
                int16_t G_left  = static_cast<int16_t>(P[-1]); // G at (y, x-1)
                int16_t G_right = static_cast<int16_t>(P[1]);  // G at (y, x+1)
                int16_t B_center = static_cast<int16_t>(P[0]); // B at (y, x)
                int16_t B_left  = static_cast<int16_t>(P[-2]); // B at (y, x-2)
                int16_t B_right = static_cast<int16_t>(P[2]);  // B at (y, x+2)

                // Compute horizontal Green estimation
                G_h = ((G_left + G_right + 1) >> 1) + ((2 * B_center - B_left - B_right + 2) >> 2);

                // Vertical estimation
                int16_t G_up    = static_cast<int16_t>(P[-raw_pitch]);       // G at (y-1, x)
                int16_t G_down  = static_cast<int16_t>(P[raw_pitch]);        // G at (y+1, x)
                int16_t B_up    = static_cast<int16_t>(P[-2 * raw_pitch]);   // B at (y-2, x)
                int16_t B_down  = static_cast<int16_t>(P[2 * raw_pitch]);    // B at (y+2, x)

                // Compute vertical Green estimation
                G_v = ((G_up + G_down + 1) >> 1) + ((2 * B_center - B_up - B_down + 2) >> 2);

                // Compute classifiers S_h and S_v for decision making
                int16_t C_center_h = B_center - G_h;
                int16_t C_left      = B_left - G_left;
                int16_t C_right     = B_right - G_right;
                int16_t S_h         = std::abs(C_center_h - C_left) + std::abs(C_center_h - C_right);

                int16_t C_center_v = B_center - G_v;
                int16_t C_up        = B_up - G_up;
                int16_t C_down      = B_down - G_down;
                int16_t S_v         = std::abs(C_center_v - C_up) + std::abs(C_center_v - C_down);

                // Decision based on classifiers
                int16_t G_est = (S_h <= S_v) ? G_h : G_v;

                // Optional Close Averaging for smoother transitions
                #ifdef ENABLE_CLOSE_AVERAGING
                if (std::abs(S_h - S_v) <= 29) {
                    G_est = (G_h + G_v + 1) >> 1;
                }
                #endif

                // Assign Blue and estimated Green values to the BGR buffer
                bgr_p0[0] = P[0]; // Blue channel
                bgr_p0[1] = saturate_cast_int16_to_uint8(G_est); // Green channel
                // Red channel will be filled in a separate function
            }

            // --- Upper Right Pixel (P1): Green Pixel ---
            {
                const uint8_t* P1 = block + 1;
                uint8_t* bgr_p1    = bgr_block + 3;

                // Green channel is directly available from the raw data
                bgr_p1[1] = P1[0];
                // Blue and Red channels will be filled in separate functions
            }

            // --- Lower Left Pixel (P2): Green Pixel ---
            {
                const uint8_t* P2 = block + raw_pitch;
                uint8_t* bgr_p2    = bgr_block + bgr_pitch;

                // Green channel is directly available from the raw data
                bgr_p2[1] = P2[0];
                // Blue and Red channels will be filled in separate functions
            }

            // --- Lower Right Pixel (P3): Red Pixel ---
            {
                const uint8_t* P3 = block + raw_pitch + 1;
                uint8_t* bgr_p3    = bgr_block + bgr_pitch + 3;

                // Initialize estimations for Green channel
                int16_t G_h = 0; // Horizontal estimation
                int16_t G_v = 0; // Vertical estimation

                // Horizontal estimation
                int16_t G_left  = static_cast<int16_t>(P3[-1]); // G at (y+1, x)
                int16_t G_right = static_cast<int16_t>(P3[1]);  // G at (y+1, x+2)
                int16_t R_center = static_cast<int16_t>(P3[0]); // R at (y+1, x+1)
                int16_t R_left  = static_cast<int16_t>(P3[-2]); // R at (y+1, x-1)
                int16_t R_right = static_cast<int16_t>(P3[2]);  // R at (y+1, x+3)

                // Compute horizontal Green estimation
                G_h = ((G_left + G_right + 1) >> 1) + ((2 * R_center - R_left - R_right + 2) >> 2);

                // Vertical estimation
                int16_t G_up    = static_cast<int16_t>(P3[-raw_pitch]);       // G at (y, x+1)
                int16_t G_down  = static_cast<int16_t>(P3[raw_pitch]);        // G at (y+2, x+1)
                int16_t R_up    = static_cast<int16_t>(P3[-2 * raw_pitch]);   // R at (y-1, x+1)
                int16_t R_down  = static_cast<int16_t>(P3[2 * raw_pitch]);    // R at (y+3, x+1)

                // Compute vertical Green estimation
                G_v = ((G_up + G_down + 1) >> 1) + ((2 * R_center - R_up - R_down + 2) >> 2);

                // Compute classifiers S_h and S_v for decision making
                int16_t C_center_h = R_center - G_h;
                int16_t C_left      = R_left - G_left;
                int16_t C_right     = R_right - G_right;
                int16_t S_h         = std::abs(C_center_h - C_left) + std::abs(C_center_h - C_right);

                int16_t C_center_v = R_center - G_v;
                int16_t C_up        = R_up - G_up;
                int16_t C_down      = R_down - G_down;
                int16_t S_v         = std::abs(C_center_v - C_up) + std::abs(C_center_v - C_down);

                // Decision based on classifiers
                int16_t G_est_p3 = (S_h <= S_v) ? G_h : G_v;

                // Optional Close Averaging for smoother transitions
                #ifdef ENABLE_CLOSE_AVERAGING
                if (std::abs(S_h - S_v) <= 27) {
                    G_est_p3 = (G_h + G_v + 1) >> 1;
                }
                #endif

                // Assign Red and estimated Green values to the BGR buffer
                bgr_p3[2] = P3[0]; // Red channel
                bgr_p3[1] = saturate_cast_int16_to_uint8(G_est_p3); // Green channel
                // Blue channel will be filled in separate functions
            }
        }
    }
}

/**
 * @brief Estimates and fills the Red and Blue channels in a BGGR Bayer pattern image.
 *
 * This function processes the partially filled BGR buffer to estimate the missing Red and
 * Blue channel values using the Menon 2007 algorithm. It iterates over 2x2 pixel blocks
 * and computes the Red and Blue values based on neighboring pixel information.
 *
 * @param bgr Pointer to the BGR data buffer (Green channel should be already filled).
 * @param bgr_pitch Number of bytes per row in the BGR data buffer.
 * @param width Width of the image in pixels (must be even).
 * @param height Height of the image in pixels (must be even).
 *
 * @pre `bgr` must not be `nullptr`.
 * @pre `width` and `height` must be positive and even.
 * @pre `bgr_pitch` must be sufficient to hold the image data.
 *
 * @note The function assumes that the Green channel has been estimated and filled
 *       before calling this function.
 */
void bggr_menon2007_rb_cpu(
    uint8_t* bgr,
    int bgr_pitch,
    int width,
    int height)
{
    // Validate input parameters
    assert(bgr != nullptr && "BGR data pointer must not be null.");
    assert(width > 0 && "Image width must be positive.");
    assert(height > 0 && "Image height must be positive.");
    assert(width % 2 == 0 && "Image width must be even.");
    assert(height % 2 == 0 && "Image height must be even.");
    assert(bgr_pitch >= width * 3 && "BGR pitch must be at least equal to image width * 3.");

    // Iterate over the image in 2x2 blocks
    for(int y = 0; y < height; y += 2) {
        for(int x = 0; x < width; x += 2) {
            // Calculate the starting index of the 2x2 block in the BGR buffer
            uint8_t* bgr_block = bgr + y * bgr_pitch + x * 3;

            // Pointers to the four pixels in the 2x2 block
            uint8_t* P0 = bgr_block;                     // Upper Left (B)
            uint8_t* P1 = bgr_block + 3;                 // Upper Right (G)
            uint8_t* P2 = bgr_block + bgr_pitch;         // Lower Left (G)
            uint8_t* P3 = bgr_block + bgr_pitch + 3;     // Lower Right (R)

            // --- Estimate Red at P0 (Blue Pixel) ---
            {
                // Neighboring Red and Green pixels
                // Ensure that we do not access out-of-bounds memory
                // This assumes that the image has been appropriately padded
                int16_t R_UR = *(P0 + 2 + 3 - bgr_pitch);   // R at (x+1, y-1)
                int16_t R_LL = *(P0 + 2 - 3 + bgr_pitch);   // R at (x-1, y+1)
                int16_t G_UR = *(P0 + 1 + 3 - bgr_pitch);   // G at (x+1, y-1)
                int16_t G_LL = *(P0 + 1 - 3 + bgr_pitch);   // G at (x-1, y+1)

                int16_t R_UL = *(P0 + 2 - 3 - bgr_pitch);   // R at (x-1, y-1)
                int16_t R_LR = *(P0 + 2 + 3 + bgr_pitch);   // R at (x+1, y+1)
                int16_t G_UL = *(P0 + 1 - 3 - bgr_pitch);   // G at (x-1, y-1)
                int16_t G_LR = *(P0 + 1 + 3 + bgr_pitch);   // G at (x+1, y+1)

                // Compute color differences for Red channel estimation
                int16_t CD_UR = R_UR - G_UR;
                int16_t CD_LL = R_LL - G_LL;
                int16_t CD_UL = R_UL - G_UL;
                int16_t CD_LR = R_LR - G_LR;

                // Horizontal and Vertical estimates
                int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
                int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

                // Gradients to determine the direction of interpolation
                int16_t Grad_h = std::abs(CD_UL - CD_LR);
                int16_t Grad_v = std::abs(CD_UR - CD_LL);

                // Decision based on gradients
                int16_t CD_est = (Grad_h <= Grad_v) ? CD_h : CD_v;

                // Optional Close Averaging for smoother transitions
                #ifdef ENABLE_CLOSE_AVERAGING
                if (std::abs(Grad_h - Grad_v) <= 26) {
                    CD_est = (CD_h + CD_v + 1) >> 1;
                }
                #endif

                // Estimate Red value
                int16_t G_center = static_cast<int16_t>(P0[1]); // Green at P0
                int16_t R_est    = G_center + CD_est;

                // Clamp and assign Red value
                P0[2] = saturate_cast_int16_to_uint8(R_est);
            }

            // --- Estimate Blue at P3 (Red Pixel) ---
            {
                // Neighboring Blue and Green pixels
                uint8_t B_UR = *(P3 + 0 + 3 - bgr_pitch);   // B at (x+1, y-1)
                uint8_t B_LL = *(P3 + 0 - 3 + bgr_pitch);   // B at (x-1, y+1)
                uint8_t G_UR = *(P3 + 1 + 3 - bgr_pitch);   // G at (x+1, y-1)
                uint8_t G_LL = *(P3 + 1 - 3 + bgr_pitch);   // G at (x-1, y+1)

                uint8_t B_UL = *(P3 + 0 - 3 - bgr_pitch);   // B at (x-1, y-1)
                uint8_t B_LR = *(P3 + 0 + 3 + bgr_pitch);   // B at (x+1, y+1)
                uint8_t G_UL = *(P3 + 1 - 3 - bgr_pitch);   // G at (x-1, y-1)
                uint8_t G_LR = *(P3 + 1 + 3 + bgr_pitch);   // G at (x+1, y+1)

                // Compute color differences for Blue channel estimation
                int16_t CD_UR = static_cast<int16_t>(B_UR) - static_cast<int16_t>(G_UR);
                int16_t CD_LL = static_cast<int16_t>(B_LL) - static_cast<int16_t>(G_LL);
                int16_t CD_UL = static_cast<int16_t>(B_UL) - static_cast<int16_t>(G_UL);
                int16_t CD_LR = static_cast<int16_t>(B_LR) - static_cast<int16_t>(G_LR);

                // Horizontal and Vertical estimates
                int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
                int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

                // Gradients to determine the direction of interpolation
                int16_t Grad_h = std::abs(CD_UL - CD_LR);
                int16_t Grad_v = std::abs(CD_UR - CD_LL);

                // Decision based on gradients
                int16_t CD_est = (Grad_h <= Grad_v) ? CD_h : CD_v;

                // Optional Close Averaging for smoother transitions
                #ifdef ENABLE_CLOSE_AVERAGING
                if (std::abs(Grad_h - Grad_v) <= 26) {
                    CD_est = (CD_h + CD_v + 1) >> 1;
                }
                #endif

                // Estimate Blue value
                int16_t G_center = static_cast<int16_t>(P3[1]); // Green at P3
                int16_t B_est    = G_center + CD_est;

                // Clamp and assign Blue value
                P3[0] = saturate_cast_int16_to_uint8(B_est);
            }

            // --- Estimate Red and Blue at P1 (Upper Right Green Pixel) ---
            {
                // --- Estimate Red at P1 ---
                // Bilinear interpolation of (R - G)
                uint8_t R_up    = *(P1 + 2 - bgr_pitch);   // R at (x+1, y-1)
                uint8_t R_down  = *(P1 + 2 + bgr_pitch);   // R at (x+1, y+1)
                uint8_t G_up    = *(P1 + 1 - bgr_pitch);   // G at (x+1, y-1)
                uint8_t G_down  = *(P1 + 1 + bgr_pitch);   // G at (x+1, y+1)

                int16_t CD_RU = static_cast<int16_t>(R_up) - static_cast<int16_t>(G_up);
                int16_t CD_RD = static_cast<int16_t>(R_down) - static_cast<int16_t>(G_down);
                int16_t CD_R  = (CD_RU + CD_RD + 1) >> 1;

                int16_t G_center = static_cast<int16_t>(P1[1]); // Green at P1
                int16_t R_est    = G_center + CD_R;

                // --- Estimate Blue at P1 ---
                // Bilinear interpolation of (B - G)
                uint8_t B_left  = *(P1 - 3 + 0); // B at (x, y)
                uint8_t B_right = *(P1 + 3 + 0); // B at (x+2, y)
                uint8_t G_left  = *(P1 - 3 + 1); // G at (x, y)
                uint8_t G_right = *(P1 + 3 + 1); // G at (x+2, y)

                int16_t CD_BL = static_cast<int16_t>(B_left) - static_cast<int16_t>(G_left);
                int16_t CD_BR = static_cast<int16_t>(B_right) - static_cast<int16_t>(G_right);
                int16_t CD_B  = (CD_BL + CD_BR + 1) >> 1;

                int16_t B_est = G_center + CD_B;

                // Clamp and assign Red and Blue values
                P1[2] = saturate_cast_int16_to_uint8(R_est); // Red channel
                P1[0] = saturate_cast_int16_to_uint8(B_est); // Blue channel
            }

            // --- Estimate Red and Blue at P2 (Lower Left Green Pixel) ---
            {
                // --- Estimate Red at P2 ---
                // Bilinear interpolation of (R - G)
                uint8_t R_left  = *(P2 - 3 + 2); // R at (x-1, y+1)
                uint8_t R_right = *(P2 + 3 + 2); // R at (x+1, y+1)
                uint8_t G_left  = *(P2 - 3 + 1); // G at (x-1, y+1)
                uint8_t G_right = *(P2 + 3 + 1); // G at (x+1, y+1)

                int16_t CD_RL = static_cast<int16_t>(R_left) - static_cast<int16_t>(G_left);
                int16_t CD_RR = static_cast<int16_t>(R_right) - static_cast<int16_t>(G_right);
                int16_t CD_R  = (CD_RL + CD_RR + 1) >> 1;

                int16_t G_center = static_cast<int16_t>(P2[1]); // Green at P2
                int16_t R_est    = G_center + CD_R;

                // --- Estimate Blue at P2 ---
                // Bilinear interpolation of (B - G)
                uint8_t B_up    = *(P2 - bgr_pitch + 0); // B at (x, y)
                uint8_t B_down  = *(P2 + bgr_pitch + 0); // B at (x, y+2)
                uint8_t G_up    = *(P2 - bgr_pitch + 1); // G at (x, y)
                uint8_t G_down  = *(P2 + bgr_pitch + 1); // G at (x, y+2)

                int16_t CD_BU = static_cast<int16_t>(B_up) - static_cast<int16_t>(G_up);
                int16_t CD_BD = static_cast<int16_t>(B_down) - static_cast<int16_t>(G_down);
                int16_t CD_B  = (CD_BU + CD_BD + 1) >> 1;

                int16_t B_est = G_center + CD_B;

                // Clamp and assign Red and Blue values
                P2[2] = saturate_cast_int16_to_uint8(R_est); // Red channel
                P2[0] = saturate_cast_int16_to_uint8(B_est); // Blue channel
            }
        }
    }
}
