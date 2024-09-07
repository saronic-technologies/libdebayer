use anyhow::Result;

use defvar::defvar;

use opencv::core::{mean, multiply, absdiff, no_array, Mat, Vec3b};
use opencv::imgcodecs::{imread, imwrite};
use opencv::prelude::*;

use libdebayer::{DebayerInputImage, DebayerAlgorithm};

use std::fs::DirEntry;
use std::io;
use std::path::Path;

defvar! { KODAK_FOLDER_PATH: String = "../../benchmark/kodak".to_string() }

fn convert_to_rggb(bgr_image: &Mat) -> Result<Mat, opencv::Error> {
    let mut bayer = Mat::new_rows_cols_with_default(
        bgr_image.rows(),
        bgr_image.cols(),
        opencv::core::CV_8UC1,
        opencv::core::Scalar::all(0.0)
    )?;

    for y in 0..bgr_image.rows() {
        for x in 0..bgr_image.cols() {
            let pixel = bgr_image.at_2d::<Vec3b>(y, x)?;
            let value = if y % 2 == 0 {
                if x % 2 == 0 {
                    pixel[2] // R (index 2 in BGR)
                } else {
                    pixel[1] // G (index 1 in BGR)
                }
            } else {
                if x % 2 == 0 {
                    pixel[1] // G (index 1 in BGR)
                } else {
                    pixel[0] // B (index 0 in BGR)
                }
            };
            *bayer.at_2d_mut::<u8>(y, x)? = value;
        }
    }

    Ok(bayer)
}

fn convert_to_bggr(bgr_image: &Mat) -> Result<Mat, opencv::Error> {
    let mut bayer = Mat::new_rows_cols_with_default(
        bgr_image.rows(),
        bgr_image.cols(),
        opencv::core::CV_8UC1,
        opencv::core::Scalar::all(0.0)
    )?;

    for y in 0..bgr_image.rows() {
        for x in 0..bgr_image.cols() {
            let pixel = bgr_image.at_2d::<Vec3b>(y, x)?;
            let value = if y % 2 == 0 {
                if x % 2 == 0 {
                    pixel[0] // B (index 0 in BGR)
                } else {
                    pixel[1] // G (index 1 in BGR)
                }
            } else {
                if x % 2 == 0 {
                    pixel[1] // G (index 1 in BGR)
                } else {
                    pixel[2] // R (index 2 in BGR)
                }
            };
            *bayer.at_2d_mut::<u8>(y, x)? = value;
        }
    }

    Ok(bayer)
}

fn calculate_psnr(original: &Mat, processed: &Mat) -> Result<f64> {
    let mut diff = Mat::default();
    absdiff(original, processed, &mut diff)?;

    let mut float_diff = Mat::default();
    diff.convert_to(&mut float_diff, opencv::core::CV_32F, 1.0, 0.0)?;
    multiply(&float_diff, &float_diff, &mut diff, 1.0, -1)?;

    let mse = mean(&diff, &no_array())?[0];
    if mse <= 1e-10 {
        return Ok(100.0);  // Indicates nearly identical images
    }

    let max_pixel_value = 255.0;
    let psnr = 20.0 * (max_pixel_value / mse.sqrt()).log10();
    
    Ok(psnr)
}

fn main() -> Result<(), anyhow::Error> {
    let kodak_path = Path::new((*KODAK_FOLDER_PATH).as_str());

    let test_files: Vec<Result<DirEntry, io::Error>> = kodak_path.read_dir()?.filter(|e| {
        if let Ok(e) = e {
            let fname = e.file_name().into_string().unwrap();
            fname.contains("kodim") && fname.contains(".png")
        } else {
            false
        }
    }).collect();

    let mut img_count = 0;
    let mut psnr_sum = 0.0f64;
    for t in test_files {
        if let Ok(t) = t {
            let fname = t.path().into_os_string().into_string().unwrap();
            println!("Processing: {fname}");

            let img = imread(fname.as_str(), opencv::imgcodecs::IMREAD_COLOR)?;

            let bayer_image = convert_to_rggb(&img)?;
            let params = opencv::core::Vector::new();
            imwrite("bayer_image.png", &bayer_image, &params);
            let mut debayer = Mat::default();
            opencv::imgproc::demosaicing(
                &bayer_image,
                &mut debayer,
                opencv::imgproc::COLOR_BayerBG2BGR_EA,
                0,
            )?;
            imwrite("debayer_opencv.png", &debayer, &params);


            let mut debayer_input_image = DebayerInputImage::try_from(bayer_image.clone())?;
            let debayer_output_image = debayer_input_image.debayer(DebayerAlgorithm::Malvar2004Rggb2Bgr)?;

            let debayer_output_mat = Mat::try_from(debayer_output_image)?;


            imwrite("test.png", &debayer_output_mat, &params);
            
            psnr_sum += calculate_psnr(&img, &debayer_output_mat)?;

            std::process::exit(1);
            img_count += 1;
        }
    }

    println!("Average PSNR: {}", psnr_sum/(img_count as f64));
    Ok(())
}
