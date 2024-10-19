/**
 * @file debayer_cpp.cpp
 * @brief CPU-based debayering implementation for converting raw Bayer images to BGR format.
 */

#include "cpu_debayer.hpp"

#define ENABLE_PARALLELISM

std::unique_ptr<ThreadPool> Debayer::thread_pool = nullptr;

// Destructor: Ensure memory is freed
Debayer::~Debayer()
{
    Free();
}

void Debayer::InitializeThreadPool()
{
    if (!thread_pool) {
        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4; // Fallback to 4 threads if detection fails
        thread_pool = std::make_unique<ThreadPool>(num_threads);
    }
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

    InitializeThreadPool();

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

#ifdef ENABLE_PARALLELISM

    // Define tile size (adjust as needed)
    const int TILE_SIZE = 64;

    // Calculate number of tiles in x and y directions
    int num_tiles_x = (raw_padded_width + TILE_SIZE - 1) / TILE_SIZE;
    int num_tiles_y = (raw_padded_height + TILE_SIZE - 1) / TILE_SIZE;

    // Vector to hold futures for green channel tasks
    std::vector<std::future<void>> green_futures;

    // Submit Green Channel Estimation tasks
    for (int ty = 0; ty < num_tiles_y; ++ty) {
        for (int tx = 0; tx < num_tiles_x; ++tx) {
            // Compute tile boundaries
            int start_x = tx * TILE_SIZE;
            int start_y = ty * TILE_SIZE;
            int end_x = std::min(start_x + TILE_SIZE, raw_padded_width);
            int end_y = std::min(start_y + TILE_SIZE, raw_padded_height);
            int tile_width = end_x - start_x;
            int tile_height = end_y - start_y;

            // Pointers to the current tile in raw and BGR data
            uint8_t* raw_tile = raw_padded_data + start_y * raw_padded_pitch + start_x;
            uint8_t* bgr_tile = bgr_padded_data + start_y * bgr_padded_pitch + start_x * 3; // 3 bytes per pixel

            // Capture necessary variables by value to avoid data races
            green_futures.emplace_back(
                thread_pool->Submit([=]() {
                    bggr_menon2007_g_cpu(
                        raw_tile,
                        raw_padded_pitch,
                        bgr_tile,
                        bgr_padded_pitch,
                        tile_width,
                        tile_height
                    );
                })
            );
        }
    }

    // Wait for all green channel tasks to complete
    for (auto &fut : green_futures) {
        fut.get();
    }

    // Vector to hold futures for red and blue channel tasks
    std::vector<std::future<void>> rb_futures;

    // Submit Red and Blue Channels Estimation tasks
    for (int ty = 0; ty < num_tiles_y; ++ty) {
        for (int tx = 0; tx < num_tiles_x; ++tx) {
            // Compute tile boundaries
            int start_x = tx * TILE_SIZE;
            int start_y = ty * TILE_SIZE;
            int end_x = std::min(start_x + TILE_SIZE, raw_padded_width);
            int end_y = std::min(start_y + TILE_SIZE, raw_padded_height);
            int tile_width = end_x - start_x;
            int tile_height = end_y - start_y;

            // Pointer to the current tile in BGR data
            uint8_t* bgr_tile = bgr_padded_data + start_y * bgr_padded_pitch + start_x * 3; // 3 bytes per pixel

            // Capture necessary variables by value to avoid data races
            rb_futures.emplace_back(
                thread_pool->Submit([=]() {
                    bggr_menon2007_rb_cpu(
                        bgr_tile,
                        bgr_padded_pitch,
                        tile_width,
                        tile_height
                    );
                })
            );
        }
    }

    // Wait for all red and blue channel tasks to complete
    for (auto &fut : rb_futures) {
        fut.get();
    }

    // Step 5: Extract the demosaiced BGR data from the padded buffer to the output image
    // Calculate output pitch (output->pitch or tightly packed)
    int output_pitch = (output->pitch != 0) ? output->pitch : output->width * 3;

    // Vector to hold futures for the copy tasks
    std::vector<std::future<void>> copy_futures;

    for (int y = 0; y < input->height; ++y) {
        copy_futures.emplace_back(
            thread_pool->Submit([=]() {
                uint8_t* src_row = bgr_padded_data + (SARONIC_DEBAYER_PAD + y) * bgr_padded_pitch + SARONIC_DEBAYER_PAD * 3;
                uint8_t* dst_row = output->bgr_data + y * output_pitch;
                std::memcpy(dst_row, src_row, input->width * 3 * sizeof(uint8_t));
            })
        );
    }

    // Wait for all copy tasks to complete
    for (auto &fut : copy_futures) {
        fut.get();
    }

#else

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

    for (int y = 0; y < input->height; ++y) {
        uint8_t* src_row = bgr_padded_data + (SARONIC_DEBAYER_PAD + y) * bgr_padded_pitch + SARONIC_DEBAYER_PAD * 3;
        uint8_t* dst_row = output->bgr_data + y * ((output->pitch != 0) ? output->pitch : output->width * 3);
        std::memcpy(dst_row, src_row, input->width * 3 * sizeof(uint8_t));
     }
#endif

    // Free the allocated padded buffers
    Free();

    return 0; // Success
}
