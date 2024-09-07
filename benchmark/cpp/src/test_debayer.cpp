#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <cstdlib>


#include <opencv2/opencv.hpp>

#include <npp.h>
#include <cuda_runtime.h>

#include "libdebayercpp/debayer_cpp.h"

using namespace std::chrono;
namespace fs = std::filesystem;


//------------------------------------------------------------------------------
// Tools

static uint64_t getUsec() {
    auto now = high_resolution_clock::now();
    return duration_cast<microseconds>(now.time_since_epoch()).count();
}

// Function to convert BGR image to RGGB Bayer pattern
cv::Mat convertToRGGB(const cv::Mat& bgr_image) {
    cv::Mat bayer(bgr_image.rows, bgr_image.cols, CV_8UC1);
    for (int y = 0; y < bgr_image.rows; ++y) {
        for (int x = 0; x < bgr_image.cols; ++x) {
            cv::Vec3b pixel = bgr_image.at<cv::Vec3b>(y, x);
            if (y % 2 == 0) {
                if (x % 2 == 0) {
                    bayer.at<uchar>(y, x) = pixel[2]; // R (index 2 in BGR)
                } else {
                    bayer.at<uchar>(y, x) = pixel[1]; // G (index 1 in BGR)
                }
            } else {
                if (x % 2 == 0) {
                    bayer.at<uchar>(y, x) = pixel[1]; // G (index 1 in BGR)
                } else {
                    bayer.at<uchar>(y, x) = pixel[0]; // B (index 0 in BGR)
                }
            }
        }
    }
    return bayer;
}

cv::Mat convertToBGGR(const cv::Mat& bgr_image) {
    cv::Mat bayer(bgr_image.rows, bgr_image.cols, CV_8UC1);
    for (int y = 0; y < bgr_image.rows; ++y) {
        for (int x = 0; x < bgr_image.cols; ++x) {
            cv::Vec3b pixel = bgr_image.at<cv::Vec3b>(y, x);
            if (y % 2 == 0) {
                if (x % 2 == 0) {
                    bayer.at<uchar>(y, x) = pixel[0]; // B (index 0 in BGR)
                } else {
                    bayer.at<uchar>(y, x) = pixel[1]; // G (index 1 in BGR)
                }
            } else {
                if (x % 2 == 0) {
                    bayer.at<uchar>(y, x) = pixel[1]; // G (index 1 in BGR)
                } else {
                    bayer.at<uchar>(y, x) = pixel[2]; // R (index 2 in BGR)
                }
            }
        }
    }
    return bayer;
}

double calculatePSNR_RB_at_BR(const cv::Mat& original, const cv::Mat& processed) {
    // Ensure the images are 3-channel (BGR) images and have even dimensions
    CV_Assert(original.channels() == 3 && processed.channels() == 3);
    CV_Assert(original.rows == processed.rows && original.cols == processed.cols);
    CV_Assert(original.rows % 2 == 0 && original.cols % 2 == 0);

    cv::Mat diff(original.rows / 2, original.cols / 2, CV_32FC2);
    double sum_squared_diff = 0.0;
    int count = 0;

    for (int y = 4; y < original.rows - 4; y += 2) {
        for (int x = 4; x < original.cols - 4; x += 2) {
            // Get the diagonal pixels (top-left and bottom-right of each 2x2 tile)
            cv::Vec3b orig_tl = original.at<cv::Vec3b>(y, x);
            cv::Vec3b proc_tl = processed.at<cv::Vec3b>(y, x);
            cv::Vec3b orig_br = original.at<cv::Vec3b>(y+1, x+1);
            cv::Vec3b proc_br = processed.at<cv::Vec3b>(y+1, x+1);

            // Calculate differences for R (index 2) and B (index 0) channels
            double diff_b = static_cast<double>(orig_tl[1]) - static_cast<double>(proc_tl[1]);
            double diff_r = static_cast<double>(orig_br[1]) - static_cast<double>(proc_br[1]);

            // Accumulate squared differences
            sum_squared_diff += diff_r * diff_r + diff_b * diff_b;
            count += 2; // We're considering 2 values per tile

            // Store the differences (for visualization if needed)
            diff.at<cv::Vec2f>(y/2, x/2) = cv::Vec2f(static_cast<float>(diff_r), static_cast<float>(diff_b));
        }
    }

    // Calculate MSE
    double mse = sum_squared_diff / count;

    std::cout << "MSE: " << mse << std::endl;
    std::cout << "Count: " << count << std::endl;
    std::cout << "Sum of squared differences: " << sum_squared_diff << std::endl;

    if (mse <= 1e-10) {
        std::cout << "Warning: Images appear to be identical or very close." << std::endl;
        return 100.0;  // Indicates nearly identical diagonal R and B values
    }

    double max_pixel_value = 255.0;
    double psnr = 20 * log10(max_pixel_value / sqrt(mse));
    return psnr;
}

double calculatePSNR(const cv::Mat& original, const cv::Mat& processed) {
    cv::Mat diff;
    cv::absdiff(original, processed, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = cv::mean(diff)[0];
    if (mse <= 1e-10) {
        return 100.0;  // Indicates nearly identical images
    }

    double max_pixel_value = 255.0;
    double psnr = 20 * log10(max_pixel_value / sqrt(mse));
    return psnr;
}


//------------------------------------------------------------------------------
// OpenCV Debayering Functions

cv::Mat debayerEA(const cv::Mat& bayer_image) {
    cv::Mat output_image;
    cv::cvtColor(bayer_image, output_image, cv::COLOR_BayerBG2BGR_EA);
    return output_image;
}

cv::Mat debayerVNG(const cv::Mat& bayer_image) {
    cv::Mat output_image;
    cv::cvtColor(bayer_image, output_image, cv::COLOR_BayerBG2BGR_VNG);
    return output_image;
}

//------------------------------------------------------------------------------
// NPP Debayering Functions

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// NPP error checking macro
#define CHECK_NPP(call) \
    do { \
        NppStatus status = call; \
        if (status != NPP_SUCCESS) { \
            std::cerr << "NPP error in " << __FILE__ << ":" << __LINE__ << ": " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

cv::Mat debayerNPP(const cv::Mat& rggb) {
    // Allocate device memory
    Npp8u* d_rggb;
    Npp8u* d_rgb;
    size_t pitch_rggb, pitch_rgb;

    CHECK_CUDA(cudaMallocPitch(&d_rggb, &pitch_rggb, rggb.cols * sizeof(Npp8u), rggb.rows));
    CHECK_CUDA(cudaMallocPitch(&d_rgb, &pitch_rgb, rggb.cols * 3 * sizeof(Npp8u), rggb.rows));

    // Copy RGGB data to device
    CHECK_CUDA(cudaMemcpy2D(d_rggb, pitch_rggb, rggb.data, rggb.step, rggb.cols * sizeof(Npp8u), rggb.rows, cudaMemcpyHostToDevice));

    // Set up NPP parameters
    NppiSize oSrcSize = {rggb.cols, rggb.rows};
    NppiRect oSrcROI = {0, 0, rggb.cols, rggb.rows};

    // Convert RGGB to RGB using NPP
    CHECK_NPP(nppiCFAToRGB_8u_C1C3R(d_rggb, pitch_rggb, oSrcSize, oSrcROI, d_rgb, pitch_rgb, NPPI_BAYER_RGGB, NPPI_INTER_UNDEFINED));

    // Allocate host memory for the result
    cv::Mat output_rgb(rggb.rows, rggb.cols, CV_8UC3);

    // Copy the result back to host
    CHECK_CUDA(cudaMemcpy2D(output_rgb.data, output_rgb.step, d_rgb, pitch_rgb, rggb.cols * 3 * sizeof(Npp8u), rggb.rows, cudaMemcpyDeviceToHost));

    // Convert RGB to BGR
    cv::Mat output_bgr;
    cv::cvtColor(output_rgb, output_bgr, cv::COLOR_RGB2BGR);

    return output_bgr;
}

std::string getEnvVar(const std::string& varName, const std::string& defaultValue) {
    char* envVar = std::getenv(varName.c_str());
    return envVar ? std::string(envVar) : defaultValue;
}

//------------------------------------------------------------------------------
// Entrypoint

int main() {
    std::string folder_path = getEnvVar("KODAK_FOLDER_PATH", "../../kodak");
    std::vector<std::string> image_files;

    // Find all Kodak images in the folder
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string filename = entry.path().filename().string();
        if (filename.substr(0, 5) == "kodim" && filename.substr(filename.length() - 4) == ".png" &&
            filename.substr(filename.length() - 8) != ".out.png") {
            image_files.push_back(entry.path().string());
        }
    }

    std::cout << "Found " << image_files.size() << " images under " << folder_path << std::endl;

    Debayer* context = new Debayer;
    if (!context) {
        std::cerr << "Error: Could not create Debayer context" << std::endl;
        return -1;
    }

    double psnr_sum = 0.0;
    double g_psnr_sum = 0.0;

    for (const auto& image_file : image_files) {
        std::cout << "Processing: " << image_file << std::endl;

        // Read the image
        cv::Mat bgr_image = cv::imread(image_file);
        if (bgr_image.empty()) {
            std::cerr << "Error: Could not read image " << image_file << std::endl;
            continue;
        }

        raw_image_t input;

        // Convert to Bayer pattern
        //cv::Mat bayer_image = convertToBGGR(bgr_image);
        //input.format = SARONIC_DEBAYER_BGGR;

        cv::Mat bayer_image = convertToRGGB(bgr_image);
        input.format = SARONIC_DEBAYER_RGGB;

        // Prepare raw_image_t
        input.raw_data = bayer_image.data;
        input.pitch = bayer_image.step;
        input.width = bayer_image.cols;
        input.height = bayer_image.rows;
        //input.algorithm = SARONIC_DEBAYER_BILINEAR;
        input.algorithm = SARONIC_DEBAYER_MALVAR2004;
        //input.algorithm = SARONIC_DEBAYER_SARONIC1;

        // Prepare bgr_image_t for output
        cv::Mat output_image(bgr_image.rows, bgr_image.cols, CV_8UC3);
        bgr_image_t output;
        output.bgr_data = output_image.data;
        output.pitch = output_image.step;
        output.width = output_image.cols;
        output.height = output_image.rows;

        uint64_t t0 = getUsec();

        int32_t r = context->Process(&input, &output);
        if (r != 0) {
            std::cerr << "Error(" << r << "): Could not process image " << image_file << std::endl;
            delete context;
            return -1;
        }

        uint64_t t1 = getUsec();
        int64_t dt = t1 - t0;

        //output_image = debayerEA(bayer_image);
        //output_image = debayerVNG(bayer_image);
        //output_image = debayerNPP(bayer_image);

        // Calculate PSNR
        double psnr = calculatePSNR(bgr_image, output_image);
        double g_psnr = calculatePSNR_RB_at_BR(bgr_image, output_image);
        psnr_sum += psnr;
        g_psnr_sum += g_psnr;

        // Save the debayered image
        std::string output_filename = image_file.substr(0, image_file.length() - 4) + ".out.png";
        cv::imwrite(output_filename, output_image);
        std::exit(1);

        std::cout << "Debayered image saved as: " << output_filename << std::endl;
        std::cout << " + PSNR: " << psnr << " dB. G-channel PSNR: " << g_psnr << " dB" << std::endl;
        std::cout << " + Debayering time: " << dt << " us" << std::endl;
    }

    std::cout << "-> Average PSNR: " << psnr_sum / image_files.size() << " dB" << std::endl;
    std::cout << "-> Average G-channel PSNR: " << g_psnr_sum / image_files.size() << " dB" << std::endl;

    delete context;

    return 0;
}
