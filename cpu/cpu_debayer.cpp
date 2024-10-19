/**
 * @file debayer_cpp.cpp
 * @brief CPU-based debayering implementation for converting raw Bayer images to BGR format.
 */

#include "cpu_debayer.hpp"

// Destructor: Ensure memory is freed
Debayer::~Debayer()
{
    Free();
}

// Helper function to round up a value to the nearest multiple of a modulus
int Debayer::RoundUp(int x, int modulus) {
    int r = x % modulus;
    if (r == 0) {
        return x;
    } else {
        return x + (modulus - r);
    }
}

// Allocate memory for padded raw and BGR images
bool Debayer::Allocate(int new_width, int new_height)
{
    // Check if the current allocation matches the requested size
    if (width == new_width && height == new_height) {
        return true;
    }

    // Free any previously allocated memory
    Free();

    // Update dimensions
    width = new_width;
    height = new_height;

    // Calculate padded dimensions aligned with kernel block size
    const int padded_width = SARONIC_DEBAYER_PAD + RoundUp(width + SARONIC_DEBAYER_PAD, KERNEL_BLOCK_SIZE);
    const int padded_height = SARONIC_DEBAYER_PAD + RoundUp(height + SARONIC_DEBAYER_PAD, KERNEL_BLOCK_SIZE);

    // Allocate raw padded data (one byte per pixel)
    raw_padded_width = padded_width;
    raw_padded_height = padded_height;
    raw_padded_pitch = raw_padded_width; // Tightly packed

    try {
        raw_padded_data = new uint8_t[raw_padded_pitch * raw_padded_height];
    } catch (const std::bad_alloc&) {
        std::cerr << "Error: Failed to allocate memory for raw_padded_data." << std::endl;
        return false;
    }

    // Initialize raw padded data to zero
    std::memset(raw_padded_data, 0, raw_padded_pitch * raw_padded_height);

    // Allocate BGR padded data (three bytes per pixel)
    bgr_padded_width = padded_width;
    bgr_padded_height = padded_height;
    bgr_padded_pitch = bgr_padded_width * 3; // BGR has 3 channels

    try {
        bgr_padded_data = new uint8_t[bgr_padded_pitch * bgr_padded_height];
    } catch (const std::bad_alloc&) {
        std::cerr << "Error: Failed to allocate memory for bgr_padded_data." << std::endl;
        Free();
        return false;
    }

    // Initialize BGR padded data to zero
    std::memset(bgr_padded_data, 0, bgr_padded_pitch * bgr_padded_height);

    return true;
}

// Free allocated memory buffers
void Debayer::Free()
{
    // Free raw padded data
    if (raw_padded_data) {
        delete[] raw_padded_data;
        raw_padded_data = nullptr;
    }

    // Free BGR padded data
    if (bgr_padded_data) {
        delete[] bgr_padded_data;
        bgr_padded_data = nullptr;
    }

    // Reset dimensions and pitches
    raw_padded_pitch = 0;
    raw_padded_width = 0;
    raw_padded_height = 0;

    bgr_padded_pitch = 0;
    bgr_padded_width = 0;
    bgr_padded_height = 0;

    width = 0;
    height = 0;
}

// Process a raw Bayer image and output a BGR image
int Debayer::Process(const raw_image_t* input, bgr_image_t* output)
{
    // Validate input pointers
    if (input == nullptr || output == nullptr) {
        std::cerr << "Error: Input or output image pointer is null." << std::endl;
        return -1;
    }

    // Validate image dimensions
    if (input->width != output->width || input->height != output->height) {
        std::cerr << "Error: Input and output image sizes do not match." << std::endl;
        return -2;
    }

    // Allocate memory with padding
    if (!Allocate(input->width, input->height)) {
        std::cerr << "Error: Failed to allocate memory for debayering." << std::endl;
        return -3;
    }

    // Calculate source pitch (input->pitch or tightly packed)
    int input_pitch = (input->pitch != 0) ? input->pitch : input->width;

    // Step 1: Copy raw data into the center of the padded raw buffer
    for (int y = 0; y < input->height; ++y) {
        const uint8_t* src_row = input->raw_data + y * input_pitch;
        uint8_t* dst_row = raw_padded_data + (SARONIC_DEBAYER_PAD + y) * raw_padded_pitch + SARONIC_DEBAYER_PAD;
        std::memcpy(dst_row, src_row, input->width * sizeof(uint8_t));
    }

    // Step 2: Pad the top and bottom edges
    padTopBottomEdges(raw_padded_data, input->width, input->height, raw_padded_pitch, SARONIC_DEBAYER_PAD);

    // Step 3: Pad the left and right edges
    padLeftRightEdges(raw_padded_data, input->width, input->height, raw_padded_pitch, SARONIC_DEBAYER_PAD);

    // At this point, raw_padded_data contains the padded raw image

    // Step 4: Estimate the Green channel
    // Note: The demosaicing functions expect the image to be in BGGR format
    // Ensure that input->format is BGGR; if not, additional handling is required
    if (input->format != SARONIC_DEBAYER_BGGR) {
        std::cerr << "Error: Only BGGR format is supported in this CPU implementation." << std::endl;
        Free();
        return -4;
    }

    // Estimate Green channel using the Menon 2007 algorithm
    bggr_menon2007_g_cpu(
        raw_padded_data,      // Pointer to padded raw data
        raw_padded_pitch,     // Pitch of the raw data
        bgr_padded_data,      // Pointer to padded BGR data
        bgr_padded_pitch,     // Pitch of the BGR data
        raw_padded_width,     // Width of the padded raw data
        raw_padded_height     // Height of the padded raw data
    );

    // Estimate Red and Blue channels using the Menon 2007 algorithm
    bggr_menon2007_rb_cpu(
        bgr_padded_data,      // Pointer to padded BGR data
        bgr_padded_pitch,     // Pitch of the BGR data
        raw_padded_width,     // Width of the padded raw data
        raw_padded_height     // Height of the padded raw data
    );

    // Step 5: Extract the demosaiced BGR data from the padded buffer to the output image
    for (int y = 0; y < input->height; ++y) {
        uint8_t* src_row = bgr_padded_data + (SARONIC_DEBAYER_PAD + y) * bgr_padded_pitch + SARONIC_DEBAYER_PAD * 3;
        uint8_t* dst_row = output->bgr_data + y * ((output->pitch != 0) ? output->pitch : output->width * 3);
        std::memcpy(dst_row, src_row, input->width * 3 * sizeof(uint8_t));
    }

    // Free the allocated padded buffers
    Free();

    return 0; // Success
}
