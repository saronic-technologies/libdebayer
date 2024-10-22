/**
 * @file debayer_cpp.h
 * @brief Header file for CPU-based debayering implementation.
 */

#ifndef DEBAYER_CPP_H
#define DEBAYER_CPP_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>

// Include the CPU kernel functions for padding and demosaicing
#include "cpu_kernel.hpp"

#include "threadpool.hpp"

/**
 * @brief Padding size for the debayering process.
 *
 * This constant defines the number of pixels to pad around the raw image
 * to handle edge cases during demosaicing.
 */
constexpr int SARONIC_DEBAYER_PAD = 2; // Example padding size

/**
 * @brief Block size for aligning image dimensions.
 *
 * This constant defines the block size to which image dimensions are
 * rounded up for optimal processing.
 */
constexpr int KERNEL_BLOCK_SIZE = 16; // Example block size

/**
 * @brief Enumeration of supported debayering algorithms.
 */
enum DebayerAlgorithm {
    SARONIC_DEBAYER_BILINEAR,    ///< Bilinear interpolation algorithm
    SARONIC_DEBAYER_MALVAR2004,  ///< Malvar 2004 algorithm
    SARONIC_DEBAYER_MENON2007     ///< Menon 2007 algorithm
};

/**
 * @brief Enumeration of supported Bayer pattern formats.
 */
enum BayerFormat {
    SARONIC_DEBAYER_RGGB, ///< RGGB Bayer pattern
    SARONIC_DEBAYER_BGGR  ///< BGGR Bayer pattern
};

/**
 * @brief Structure representing a raw Bayer image.
 */
struct raw_image_t {
    int width = -1;          ///< Width of the raw image in pixels
    int height = -1;         ///< Height of the raw image in pixels
    uint8_t* raw_data = nullptr;  ///< Pointer to raw Bayer data
    int pitch = 0;          ///< Number of bytes per row (0 indicates tightly packed)
    int algorithm = 0;      ///< Debayering algorithm (e.g., BILINEAR, MALVAR2004, MENON2007)
    int format = 0;         ///< Bayer pattern format (e.g., RGGB, BGGR)
};

/**
 * @brief Structure representing a BGR image.
 */
struct bgr_image_t {
    int width = -1;          ///< Width of the BGR image in pixels
    int height = -1;         ///< Height of the BGR image in pixels
    uint8_t* bgr_data = nullptr;  ///< Pointer to BGR data
    int pitch = 0;          ///< Number of bytes per row (0 indicates tightly packed)
};

/**
 * @brief Class responsible for performing debayering on CPU.
 *
 * This class handles memory allocation, image padding, and demosaicing
 * to convert raw Bayer images to BGR format.
 */
class Debayer {
public:
    /**
     * @brief Destructor: Ensures that allocated memory is freed.
     */
    ~Debayer();

    /**
     * @brief Allocates memory for padded raw and BGR images.
     *
     * This function allocates memory buffers with additional padding to handle
     * edge pixels during the demosaicing process. It ensures that the padded
     * dimensions are aligned with the kernel block size for optimal processing.
     *
     * @param width  Width of the original raw image in pixels.
     * @param height Height of the original raw image in pixels.
     * @return true  If allocation is successful.
     * @return false If allocation fails.
     */
    bool Allocate(int width, int height);

    /**
     * @brief Frees allocated memory buffers.
     *
     * This function releases the memory allocated for padded raw and BGR images.
     * It should be called before reallocating memory or upon destruction of the
     * Debayer object to prevent memory leaks.
     */
    void Free();

    /**
     * @brief Processes a raw Bayer image and outputs a BGR image.
     *
     * This function performs the following steps:
     * 1. Validates input and output image dimensions.
     * 2. Allocates memory with padding if necessary.
     * 3. Copies and pads the raw image data.
     * 4. Estimates the Green channel.
     * 5. Estimates the Red and Blue channels.
     * 6. Copies the demosaiced BGR data to the output image, excluding padding.
     *
     * @param input  Pointer to the input raw Bayer image.
     * @param output Pointer to the output BGR image.
     * @return int   0 on success, negative error codes on failure.
     */
    int Process(const raw_image_t* input, bgr_image_t* output);

private:
    // Padded raw image data
    uint8_t* raw_padded_data = nullptr;
    int raw_padded_pitch = -1;
    int raw_padded_width = -1;
    int raw_padded_height = -1;

    // Padded BGR image data
    uint8_t* bgr_padded_data = nullptr;
    int bgr_padded_pitch = -1;
    int bgr_padded_width = -1;
    int bgr_padded_height = -1;

    // Original image dimensions
    int width = -1;
    int height = -1;

    // Static thread pool instance
    static std::unique_ptr<ThreadPool> thread_pool;

    // Initialize the thread pool (called once)
    static void InitializeThreadPool();

    /**
     * @brief Helper function to round up a value to the nearest multiple of a modulus.
     *
     * @param x        The value to be rounded up.
     * @param modulus  The modulus to which x should be aligned.
     * @return int     The rounded-up value.
     */
    int RoundUp(int x, int modulus);
};

#endif // DEBAYER_CPP_H
