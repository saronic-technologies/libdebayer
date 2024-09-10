use anyhow::Result;

use defvar::defvar;

use opencv::core::{mean, multiply, absdiff, no_array, Mat, Vec2f, Vec3b};
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
            let value = {
                if y % 2 == 0 {
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

fn calculate_psnr_rb_at_br(original: &Mat, processed: &Mat) -> Result<f64> {
    // Ensure the images are 3-channel (BGR) images and have even dimensions
    assert_eq!(original.channels(), 3);
    assert_eq!(processed.channels(), 3);
    assert_eq!(original.rows(), processed.rows());
    assert_eq!(original.cols(), processed.cols());
    assert_eq!(original.rows() % 2, 0);
    assert_eq!(original.cols() % 2, 0);

    let mut diff = Mat::new_rows_cols_with_default(
        original.rows() / 2,
        original.cols() / 2,
        opencv::core::CV_32FC2,
        opencv::core::Scalar::default(),
    )?;

    let mut sum_squared_diff = 0.0;
    let mut count = 0;

    for y in (4..original.rows() - 4).step_by(2) {
        for x in (4..original.cols() - 4).step_by(2) {
            // Get the diagonal pixels (top-left and bottom-right of each 2x2 tile)
            let orig_tl = original.at_2d::<Vec3b>(y, x)?;
            let proc_tl = processed.at_2d::<Vec3b>(y, x)?;
            let orig_br = original.at_2d::<Vec3b>(y + 1, x + 1)?;
            let proc_br = processed.at_2d::<Vec3b>(y + 1, x + 1)?;

            // Calculate differences for R (index 2) and B (index 0) channels
            let diff_b = orig_tl[1] as f64 - proc_tl[1] as f64;
            let diff_r = orig_br[1] as f64 - proc_br[1] as f64;

            // Accumulate squared differences
            sum_squared_diff += diff_r * diff_r + diff_b * diff_b;
            count += 2; // We're considering 2 values per tile

            // Store the differences (for visualization if needed)
            *diff.at_2d_mut::<Vec2f>(y / 2, x / 2)? = Vec2f::from([diff_r as f32, diff_b as f32]);
        }
    }

    // Calculate MSE
    let mse = sum_squared_diff / count as f64;

    println!("MSE: {}", mse);
    println!("Count: {}", count);
    println!("Sum of squared differences: {}", sum_squared_diff);

    if mse <= 1e-10 {
        println!("Warning: Images appear to be identical or very close.");
        Ok(100.0) // Indicates nearly identical diagonal R and B values
    } else {
        let max_pixel_value = 255.0;
        let psnr = 20.0 * (max_pixel_value / mse.sqrt()).log10();
        Ok(psnr)
    }
}


fn main() -> Result<(), anyhow::Error> {
    let kodak_path = Path::new((*KODAK_FOLDER_PATH).as_str());

    let test_files: Vec<Result<DirEntry, io::Error>> = kodak_path.read_dir()?.filter(|e| {
        if let Ok(e) = e {
            let fname = e.file_name().into_string().unwrap();
            fname.contains("kodim") && fname.contains(".png") && (!fname.ends_with(".out.png"))
        } else {
            false
        }
    }).collect();

    let mut img_count = 0;
    let mut psnr_sum = 0.0f64;
    let mut g_psnr_sum = 0.0f64;
    
    for t in test_files {
        if let Ok(t) = t {
            let fname = t.path().into_os_string().into_string().unwrap();
            println!("Processing: {fname}");

            let img = imread(fname.as_str(), opencv::imgcodecs::IMREAD_COLOR)?;

            let bayer_image = convert_to_rggb(&img)?;

            let mut debayer_input_image = DebayerInputImage::try_from(&bayer_image)?;
            let debayer_output_image = debayer_input_image.debayer(DebayerAlgorithm::Malvar2004Rggb2Bgr)?;

            let debayer_output_mat = Mat::try_from(debayer_output_image)?;

            let psnr = calculate_psnr(&img, &debayer_output_mat)?;
            let g_psnr = calculate_psnr_rb_at_br(&img, &debayer_output_mat)?;
            println!("psnr: {psnr}");
            println!("g_psnr: {g_psnr}");
            
            psnr_sum += psnr;
            g_psnr_sum += g_psnr;

            img_count += 1;
        }
    }

    println!("Average PSNR: {} dB", psnr_sum/(img_count as f64));
    println!("Average G-Channel PSNR: {} dB", g_psnr_sum/(img_count as f64));
    Ok(())
}
