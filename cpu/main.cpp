/**
 * @file main.cpp
 * @brief Main application for CPU-based debayering using OpenCV.
 */

#include "debayer_cpp.h"      // Include the Debayer class header
#include "cpu_kernel.hpp"     // Include the CPU kernel functions

#include <opencv2/opencv.hpp> // OpenCV for image processing
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>

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

int main(int argc, char* argv[]) {
    // Check for proper usage
    if (argc != 3) {
        std::cerr << "Usage: ./debayer_cpu <input_image> <output_image>" << std::endl;
        return -1;
    }

    std::string input_filename = argv[1];
    std::string output_filename = argv[2];

    // Step 1: Read the input image using OpenCV
    cv::Mat input_image = cv::imread(input_filename, cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Error: Unable to read input image: " << input_filename << std::endl;
        return -2;
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
    Debayer debayer;
    int result = debayer.Process(&raw_image, &bgr_image);
    if (result != 0) {
        std::cerr << "Error: Debayering process failed with error code: " << result << std::endl;
        return -3;
    }

    // Step 5: Compute PSNR between the original and debayered images
    double psnr_value = computePSNR(input_image, output_image);
    if (psnr_value < 0) {
        std::cerr << "Error: PSNR computation failed." << std::endl;
        return -4;
    }

    std::cout << "PSNR between original and debayered image: " << psnr_value << " dB" << std::endl;

    // Step 6: Write the output image to a PNG file
    if (!cv::imwrite(output_filename, output_image)) {
        std::cerr << "Error: Failed to write output image to: " << output_filename << std::endl;
        return -5;
    }

    std::cout << "Debayered image successfully saved to: " << output_filename << std::endl;

    return 0;
}
