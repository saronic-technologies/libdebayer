/**
 * @file main.cpp
 * @brief Main application for CPU-based debayering of multiple images using OpenCV.
 */

#include "cpu_debayer.hpp"    // Include the Debayer class header
#include "cpu_kernel.hpp"     // Include the CPU kernel functions

#include <opencv2/opencv.hpp> // OpenCV for image processing
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>             // Include chrono for timing
#include <filesystem>         // Include filesystem for directory operations
#include <algorithm>          // For std::transform, std::sort
#include <string>

namespace fs = std::filesystem;

/**
 * @brief Simulates a BGGR Bayer pattern CFA from a BGR image.
 *
 * This function takes a BGR image and masks it to create a raw CFA image
 * following the BGGR Bayer pattern. Pixels not corresponding to the BGGR
 * pattern are set to zero.
 *
 * @param bgr_input The input BGR image.
 * @return std::vector<uint8_t> The simulated raw Bayer image in BGGR pattern.
 */
std::vector<uint8_t> simulateBGGR(const cv::Mat& bgr_input) {
    // Ensure the input image is in BGR format
    assert(bgr_input.type() == CV_8UC3 && "Input image must be of type CV_8UC3 (BGR).");

    int width = bgr_input.cols;
    int height = bgr_input.rows;
    std::vector<uint8_t> raw_bayer(width * height, 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Determine the Bayer pattern position
            // BGGR pattern:
            // B G B G ...
            // G R G R ...
            // B G B G ...
            // G R G R ...
            uint8_t pixel = 0;
            cv::Vec3b bgr_pixel = bgr_input.at<cv::Vec3b>(y, x);
            if (y % 2 == 0) { // Even rows
                if (x % 2 == 0) { // Even columns - Blue
                    pixel = bgr_pixel[0];
                } else { // Odd columns - Green
                    pixel = bgr_pixel[1];
                }
            } else { // Odd rows
                if (x % 2 == 0) { // Even columns - Green
                    pixel = bgr_pixel[1];
                } else { // Odd columns - Red
                    pixel = bgr_pixel[2];
                }
            }
            raw_bayer[y * width + x] = pixel;
        }
    }

    return raw_bayer;
}

/**
 * @brief Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.
 *
 * This function calculates the PSNR value, which is a measure of the similarity
 * between two images. Higher PSNR values indicate greater similarity.
 *
 * @param original The original image.
 * @param compared The image to compare against the original.
 * @return double The PSNR value in decibels (dB). Returns -1 if images have different sizes.
 */
double computePSNR(const cv::Mat& original, const cv::Mat& compared) {
    if (original.rows != compared.rows || original.cols != compared.cols || original.type() != compared.type()) {
        std::cerr << "Error: Images must have the same dimensions and type for PSNR computation." << std::endl;
        return -1.0;
    }

    cv::Mat s1;
    cv::absdiff(original, compared, s1);       // |original - compared|
    s1.convertTo(s1, CV_32F);                  // Convert to float
    s1 = s1.mul(s1);                            // (original - compared)^2

    double mse = cv::sum(s1)[0] / (double)(original.total() * original.channels());
    if (mse == 0) {
        return INFINITY;
    }

    double psnr = 10.0 * std::log10((255 * 255) / mse);
    return psnr;
}

/**
 * @brief Checks if a file has an image extension.
 *
 * @param path The file path.
 * @return true If the file has an image extension.
 * @return false Otherwise.
 */
bool isImageFile(const fs::path& path) {
    static const std::vector<std::string> image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".ppm", ".pgm"
    };

    if (!path.has_extension()) {
        return false;
    }

    std::string ext = path.extension().string();
    // Convert extension to lowercase for case-insensitive comparison
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(image_extensions.begin(), image_extensions.end(), ext) != image_extensions.end();
}

int main(int argc, char* argv[]) {
    // Check for proper usage
    if (argc != 3) {
        std::cerr << "Usage: ./debayer_cpu <input_directory> <output_directory>" << std::endl;
        return -1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    // Verify that the input directory exists and is a directory
    fs::path input_path(input_dir);
    if (!fs::exists(input_path) || !fs::is_directory(input_path)) {
        std::cerr << "Error: Input directory does not exist or is not a directory: " << input_dir << std::endl;
        return -2;
    }

    // Create the output directory if it does not exist
    fs::path output_path(output_dir);
    if (!fs::exists(output_path)) {
        try {
            fs::create_directories(output_path);
            std::cout << "Created output directory: " << output_dir << std::endl;
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error: Failed to create output directory: " << e.what() << std::endl;
            return -3;
        }
    } else if (!fs::is_directory(output_path)) {
        std::cerr << "Error: Output path exists and is not a directory: " << output_dir << std::endl;
        return -4;
    }

    // Initialize Debayer class
    Debayer debayer;

    // Vector to store debayering times (in microseconds)
    std::vector<double> debayering_times_us;

    // Iterate over each file in the input directory
    for (const auto& entry : fs::directory_iterator(input_path)) {
        if (entry.is_regular_file() && isImageFile(entry.path())) {
            fs::path file_path = entry.path();
            std::string filename = file_path.filename().string();
            std::cout << "Processing image: " << filename << std::endl;

            // Step 1: Read the input image using OpenCV
            cv::Mat input_image = cv::imread(file_path.string(), cv::IMREAD_COLOR);
            if (input_image.empty()) {
                std::cerr << "Error: Unable to read image: " << file_path << std::endl;
                continue; // Skip to the next image
            }

            // Step 2: Simulate BGGR Bayer pattern CFA
            std::vector<uint8_t> raw_bayer = simulateBGGR(input_image);

            // Create raw_image_t structure
            raw_image_t raw_image;
            raw_image.width = input_image.cols;
            raw_image.height = input_image.rows;
            raw_image.raw_data = raw_bayer.data();
            raw_image.pitch = input_image.cols; // Tightly packed
            raw_image.algorithm = SARONIC_DEBAYER_MENON2007; // Choose the debayering algorithm
            raw_image.format = SARONIC_DEBAYER_BGGR; // BGGR Bayer pattern

            // Step 3: Prepare the output BGR image
            cv::Mat output_image(input_image.rows, input_image.cols, CV_8UC3, cv::Scalar(0, 0, 0));

            // Create bgr_image_t structure
            bgr_image_t bgr_image;
            bgr_image.width = output_image.cols;
            bgr_image.height = output_image.rows;
            bgr_image.bgr_data = output_image.data;
            bgr_image.pitch = output_image.cols * 3; // Tightly packed

            // Step 4: Perform debayering using the CPU Debayer class
            // Start timing before debayering process
            auto start_time = std::chrono::high_resolution_clock::now();

            int result = debayer.Process(&raw_image, &bgr_image);

            // End timing after debayering process
            auto end_time = std::chrono::high_resolution_clock::now();

            if (result != 0) {
                std::cerr << "Error: Debayering process failed for image " << filename
                          << " with error code: " << result << std::endl;
                continue; // Skip to the next image
            }

            // Calculate the duration in nanoseconds
            auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            // Convert to microseconds for better readability
            double duration_us = static_cast<double>(duration_ns) / 1000.0;
            debayering_times_us.push_back(duration_us);

            // Step 5: Compute PSNR between the original and debayered images
            double psnr_value = computePSNR(input_image, output_image);
            if (psnr_value < 0) {
                std::cerr << "Error: PSNR computation failed for image " << filename << std::endl;
                continue; // Skip to the next image
            }

            // Report the PSNR and debayering time
            std::cout << "PSNR for " << filename << ": " << psnr_value << " dB" << std::endl;
            std::cout << "Debayering time for " << filename << ": " << duration_us << " microseconds" << std::endl;

            // Step 6: Write the output image to the output directory
            fs::path output_file = output_path / filename;
            if (!cv::imwrite(output_file.string(), output_image)) {
                std::cerr << "Error: Failed to write output image to: " << output_file << std::endl;
                continue; // Skip to the next image
            }

            std::cout << "Debayered image saved to: " << output_file << std::endl;
            std::cout << "---------------------------------------------" << std::endl;
        }
    }

    // After processing all images, compute and report the median debayering time
    if (!debayering_times_us.empty()) {
        // Create a copy of the times vector for sorting
        std::vector<double> sorted_times = debayering_times_us;
        std::sort(sorted_times.begin(), sorted_times.end());

        double median_time = 0.0;
        size_t n = sorted_times.size();
        if (n % 2 == 1) {
            // Odd number of elements
            median_time = sorted_times[n / 2];
        } else {
            // Even number of elements
            median_time = (sorted_times[n / 2 - 1] + sorted_times[n / 2]) / 2.0;
        }

        std::cout << "Median debayering time across " << n << " images: " << median_time << " microseconds" << std::endl;
    } else {
        std::cout << "No images were processed. Median debayering time cannot be computed." << std::endl;
    }

    std::cout << "Processing completed." << std::endl;
    return 0;
}
