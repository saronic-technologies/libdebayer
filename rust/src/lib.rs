use libdebayer_sys::*;

use opencv::boxed_ref::BoxedRef;
use opencv::core::{Mat, MatTraitConst};
use opencv::prelude::*;

use std::ffi::c_void;

use thiserror::Error;

fn round_up(x: usize, modulus: usize) -> usize {
    let r = x % modulus;
    if r == 0 {
        x
    } else {
        x + (modulus - r)
    }
}


#[derive(Error, Debug)]
pub enum CudaError {
    #[error("cudaMalloc failed with error code: {0}")]
    CudaMallocFailed(u32),
    #[error("cudaMemset2D failed with error code: {0}")]
    CudaMemset2DFailed(u32),
    #[error("cudaMemcpy2DAsync failed with error code: {0}")]
    CudaMemcpy2DAsyncFailed(u32),
    #[error("cudaStreamCreate failed with error code: {0}")]
    CudaStreamCreateFailed(u32),
    #[error("cudaStreamSynchronize failed with error code: {0}")]
    CudaStreamSynchronizeFailed(u32),
}

#[derive(Error, Debug)]
pub enum DebayerError {
    #[error("OpenCV error: ")]
    Opencv(#[from] opencv::error::Error),
    #[error("CUDA error: ")]
    Cuda(#[from] CudaError),
}

pub enum DebayerAlgorithm {
    BilinearRggb2Bgr,
    Malvar2004Rggb2Bgr,
    Malvar2004Bggr2Bgr,
    Saronic1Rggb2Bgr,
}

pub enum DebayerImageType {
    Input,
    Output,
}

struct CudaImage {
    pub width: usize,
    pub height: usize,
    pub pitch: usize,
    pub raw_data: *mut c_void,
}

impl CudaImage {
    fn new(width: usize, height: usize, width_multiplier: usize) -> Result<CudaImage, CudaError> {
        let padded_width = (SARONIC_DEBAYER_PAD as usize + round_up(width + SARONIC_DEBAYER_PAD as usize, KERNEL_BLOCK_SIZE as usize)) * width_multiplier;
        let padded_height = SARONIC_DEBAYER_PAD as usize + round_up(height + SARONIC_DEBAYER_PAD as usize, KERNEL_BLOCK_SIZE as usize);

        let mut pitch = 0usize;
        unsafe {
            let mut raw_data: *mut c_void = std::ptr::null_mut();

            let ret = cudaMallocPitch(&mut raw_data as *mut *mut c_void, &mut pitch as *mut usize, padded_width, padded_height);
            if ret != cudaError::cudaSuccess {
                return Err(CudaError::CudaMallocFailed(ret as u32))
            }

            let ret = cudaMemset2D(raw_data, pitch, 0, padded_width, padded_height);
            if ret != cudaError::cudaSuccess {
                return Err(CudaError::CudaMemset2DFailed(ret as u32))
            }

            Ok(CudaImage{
                width,
                height,
                pitch,
                raw_data,
            })
        }
    }
}

impl Drop for CudaImage {
    fn drop(&mut self) {
        unsafe { let _ = cudaFree(self.raw_data); }
    }
}

pub struct DebayerInputImage {
    img: CudaImage,
    stream: cudaStream_t,
}

pub struct DebayerOutputImage {
    img: CudaImage,
    stream: cudaStream_t,
}

impl DebayerInputImage {
    // pub fn new(width: usize, )


    /// Launches the debayering kernel on the GPU and uses the stream
    /// created by the `TryFrom`/constructor.
    pub fn debayer(&mut self, algorithm: DebayerAlgorithm) -> Result<DebayerOutputImage, DebayerError> {
        unsafe {
            debayer_mirror_image(self.stream, self.img.width as i32, self.img.height as i32, self.img.pitch, self.img.raw_data as *mut u8)
        }

        let output_image = CudaImage::new(self.img.width, self.img.height, 3)?;
        
        match algorithm {
            DebayerAlgorithm::BilinearRggb2Bgr => {
                unsafe {
                    debayer_rggb2bgr_bilinear(self.stream, self.img.width as i32, self.img.height as i32, self.img.pitch, output_image.pitch, self.img.raw_data as *mut u8, output_image.raw_data as *mut u8);
                }
            },
            DebayerAlgorithm::Malvar2004Rggb2Bgr => {
                unsafe {
                    debayer_rggb2bgr_malvar2004(self.stream, self.img.width as i32, self.img.height as i32, self.img.pitch, output_image.pitch, self.img.raw_data as *mut u8, output_image.raw_data as *mut u8);
                }
            },
            DebayerAlgorithm::Malvar2004Bggr2Bgr => {
                unsafe {
                    debayer_bggr2bgr_malvar2004(self.stream, self.img.width as i32, self.img.height as i32, self.img.pitch, output_image.pitch, self.img.raw_data as *mut u8, output_image.raw_data as *mut u8);
                }
            },
            DebayerAlgorithm::Saronic1Rggb2Bgr => {
                unsafe {
                    debayer_rggb2bgr_saronic1(self.stream, self.img.width as i32, self.img.height as i32, self.img.pitch, output_image.pitch, self.img.raw_data as *mut u8, output_image.raw_data as *mut u8);
                }
            }
        }

        Ok(DebayerOutputImage {
            img: output_image,
            stream: self.stream
        })
    }
}

impl TryFrom<DebayerOutputImage> for Mat {
    type Error = DebayerError;

    /// Copies the image from the GPU into an OpenCV Mat. Note this
    /// will block until the cudaStreamSynchronize completes.
    fn try_from(image: DebayerOutputImage) -> Result<Mat, Self::Error> {
        unsafe {
            let mut output_img = Mat::new_rows_cols_with_default(image.img.height as i32, image.img.width as i32, opencv::core::CV_8UC3, opencv::core::Scalar::all(0.0))?;
            let img_data = output_img.data_mut();

            let img_data_ptr = img_data as *mut c_void;

            let img_pitch = {
                let s = output_img.step1(0)?;
                if s != 0 {
                    s
                } else {
                    image.img.width * 3
                }
            };

            let raw_cuda_ptr = (image.img.raw_data as *mut u8).add((SARONIC_DEBAYER_PAD as usize) * image.img.pitch).add(SARONIC_DEBAYER_PAD as usize * 3);

            let ret = cudaMemcpy2DAsync(img_data_ptr, img_pitch, raw_cuda_ptr as *mut c_void, image.img.pitch, image.img.width * 3, image.img.height, cudaMemcpyKind::cudaMemcpyDeviceToHost, image.stream);

            if ret != cudaError::cudaSuccess {
                return Err(Self::Error::Cuda(CudaError::CudaMemcpy2DAsyncFailed(ret)));
            }

            let ret = cudaStreamSynchronize(image.stream);

            if ret != cudaError::cudaSuccess {
                return Err(Self::Error::Cuda(CudaError::CudaStreamSynchronizeFailed(ret)));
            }            

            Ok(output_img)
        }
    }
}

fn cv_to_debayer<T: MatTraitConst>(image: &T) -> Result<DebayerInputImage, DebayerError> {
    let width = image.cols() as usize;
    let height = image.rows() as usize;

    let mut stream: cudaStream_t;
    // Allocate
    unsafe {
        stream = std::ptr::null_mut();
        let ret = cudaStreamCreate(&mut stream as *mut cudaStream_t);
        if ret != cudaError::cudaSuccess {
            return Err(DebayerError::Cuda(CudaError::CudaStreamCreateFailed(ret as u32)));
        }
    }

    let input_image = CudaImage::new(width, height, 1)?;

    let input_pitch = {
        let s = image.step1(0)?;
        if s != 0 {
            s
        } else {
            width
        }            
    };

    unsafe {
        let raw_cuda_ptr = (input_image.raw_data as *mut u8).add((SARONIC_DEBAYER_PAD as usize) * input_image.pitch).add(SARONIC_DEBAYER_PAD as usize);


        let image_data_ptr = image.data() as *const c_void;
        
        let ret = cudaMemcpy2DAsync(raw_cuda_ptr as *mut c_void, input_image.pitch, image_data_ptr, input_pitch, width, height, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

        if ret != cudaError::cudaSuccess {
            return Err(DebayerError::Cuda(CudaError::CudaMemcpy2DAsyncFailed(ret as u32)));
        }
    }

    Ok(DebayerInputImage{
        img: input_image,
        stream
    })
}

impl TryFrom<&Mat> for DebayerInputImage {
    type Error = DebayerError;

    /// Copies the image from an OpenCV Mat on to the GPU. This also
    /// creates the cudaStream the cudaMemcpy will use and the
    /// debayering kernel can use.
    fn try_from(image: &Mat) -> Result<DebayerInputImage, Self::Error> {
        let r = cv_to_debayer(image);
        r
    }
}

impl TryFrom<&BoxedRef<'_, Mat>> for DebayerInputImage {
    type Error = DebayerError;

    /// Copies the image from an OpenCV Mat on to the GPU. This also
    /// creates the cudaStream the cudaMemcpy will use and the
    /// debayering kernel can use.
    fn try_from(image: &BoxedRef<'_, Mat>) -> Result<DebayerInputImage, Self::Error> {
        cv_to_debayer(image)
    }
}
