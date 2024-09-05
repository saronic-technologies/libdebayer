#ifndef DEBAYER_H
#define DEBAYER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define CPP_ONLY(x) x
#else
#define CPP_ONLY(x)
#endif


enum {
    SARONIC_DEBAYER_RGGB = 1,
    SARONIC_DEBAYER_BGGR = 2,
    SARONIC_DEBAYER_GRBG = 3,
    SARONIC_DEBAYER_GBRG = 4,
};

enum {
    SARONIC_DEBAYER_BILINEAR = 1,
    SARONIC_DEBAYER_MALVAR2004 = 2,
    SARONIC_DEBAYER_SARONIC1 = 3,
};

typedef struct raw_image_ {
    const uint8_t* raw_data CPP_ONLY(= nullptr);
    int32_t pitch CPP_ONLY(= 0); // default: width
    int32_t width CPP_ONLY(= 0);
    int32_t height CPP_ONLY(= 0);
    int32_t format CPP_ONLY(= SARONIC_DEBAYER_RGGB);
    int32_t algorithm CPP_ONLY(= SARONIC_DEBAYER_BILINEAR);
} raw_image_t;

typedef struct bgr_image_ {
    uint8_t* bgr_data CPP_ONLY(= nullptr);
    int32_t pitch CPP_ONLY(= 0); // default: width * 3
    int32_t width CPP_ONLY(= 0);
    int32_t height CPP_ONLY(= 0);
} bgr_image_t;

class Debayer {
public:
    ~Debayer() {
        Free();
    }

    int32_t Process(const raw_image_t* input, const bgr_image_t* output);

protected:
    bool Allocate(int width, int height);
    void Free();

    int32_t cuda_width = -1, cuda_height = -1;

    cudaStream_t stream = nullptr;

    uint8_t* raw_cuda_data = nullptr;
    size_t raw_cuda_pitch = 0;

    uint8_t* bgr_cuda_data = nullptr;
    size_t bgr_cuda_pitch = 0;
};

#ifdef __cplusplus
} // extern "C"
#endif

#endif // DEBAYER_H
