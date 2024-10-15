/**
 * @file cpu_kernel.hpp
 * @brief Header file for CPU-based kernel functions used in image processing, including image padding and Bayer pattern demosaicing.
 */

#ifndef CPU_KERNEL_HPP
#define CPU_KERNEL_HPP

#include <cstdint>

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
void padTopBottomEdges(uint8_t* data, int width, int height, int pitch, int pad);

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
void padLeftRightEdges(uint8_t* data, int width, int height, int pitch, int pad);

/**
 * @brief Clamps a 16-bit signed integer to an 8-bit unsigned integer range [0, 255].
 *
 * This inline helper function ensures that the input value does not exceed the
 * bounds of an 8-bit unsigned integer.
 *
 * @param x The 16-bit signed integer to clamp.
 * @return The clamped 8-bit unsigned integer.
 */
inline uint8_t saturate_cast_int16_to_uint8(int16_t x);

/**
 * @brief Estimates and fills the Green channel in a BGGR Bayer pattern image using the Menon 2007 algorithm.
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
    int height);

/**
 * @brief Estimates and fills the Red and Blue channels in a BGGR Bayer pattern image using the Menon 2007 algorithm.
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
    int height);

#endif // CPU_KERNEL_HPP
