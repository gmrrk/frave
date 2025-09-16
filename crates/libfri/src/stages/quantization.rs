use num::traits::{float::FloatCore, real::Real};

use crate::{encoder::EncoderOpts, fractal::BASE_FRAC_DEPTH, stages::wavelet_transform::WaveletImage};

fn get_quantization_matrix(quality: u8, channel: usize) -> [f32; BASE_FRAC_DEPTH] {
    let quantization_step = 2.0.powf(BASE_FRAC_DEPTH as f32 - 0.5) * (1. + 8. / 2.0.powf(11.)) / (quality as f32 / 50.);
    match quality {
        100.. => [1.0; BASE_FRAC_DEPTH],
        0..100 => if channel == 0 {
            [
                1.0,
                1.0,
                1.0,
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 2_f32)).clamp(1.,255.),
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 3_f32)).clamp(1.,255.),
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 5_f32)).clamp(1.,255.),
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 5_f32)).clamp(1.,255.),
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 6_f32)).clamp(1.,255.),
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 6_f32)).clamp(1.,255.),
            ]
        } else {
            [
                1.0,
                1.0,
                1.0,
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 3_f32)).clamp(1.,255.),
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 3_f32)).clamp(1.,255.),
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 5_f32)).clamp(1.,255.),
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 5_f32)).clamp(1.,255.),
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 7_f32)).clamp(1.,255.),
                (quantization_step / 2.0.powf(BASE_FRAC_DEPTH as f32 + 1. - 7_f32)).clamp(1.,255.),
            ]
        }
    }
}

fn get_quantization_index(coefficient: i32, quantization_step: f32, quality: u8) -> i32 {
    let nz = 1. - 2. * (100 - quality) as f32  / 100.;

    (coefficient as f32/quantization_step).round() as i32

    //if quantization_step == 1.0 {
    //  coefficient
    //}
    //else if (coefficient.abs() as f32) < -nz * quantization_step {
    //    0
    //} else {
    //    coefficient.signum()
    //        * ((coefficient.abs() as f32 + (nz * quantization_step)) / quantization_step).floor()
    //            as i32
    //}
}

fn get_dequantized_value(quantization_index: i32, quantization_step: f32, quality: u8) -> i32 {
    (quantization_index as f32*quantization_step).round() as i32
    //let nz = 1. - 2. * (100 - quality) as f32 / 100.;
    //if quantization_step == 1.0 {
    //    quantization_index
    //} else if quantization_index == 0 {
    //    0
    //} else {
    //    quantization_index.signum()
    //        * ((quantization_index.abs() as f32 - nz) * quantization_step) as i32
    //}
}
pub fn encode(
    mut image: WaveletImage,
    encoder_config: &EncoderOpts,
) -> Result<WaveletImage, String> {
    for fractal in image.complex_plane.fractals.iter_mut() {
        for (channel, channel_coef) in fractal.coefficients.iter_mut().enumerate() {
            let quantization_matrix =
                get_quantization_matrix(image.metadata.quality, channel); 
            for level in 0..9 {
                for i in (1 << level)..1 << (level + 1) {
                    if let Some(coefficient) = channel_coef.get_mut(i) {
                        if coefficient.is_some() {
                            *coefficient = Some(get_quantization_index(
                                coefficient.unwrap(),
                                quantization_matrix[level as usize],
                                image.metadata.quality,
                            ))
                        }
                    }
                }
            }
        }
    }

    Ok(image)
}

pub fn decode(
    mut image: WaveletImage,
    quantization_matrix: &[Option<i32>; 9],
) -> Result<WaveletImage, String> {
    for fractal in image.complex_plane.fractals.iter_mut() {
        for (channel, channel_coef) in fractal.coefficients.iter_mut().enumerate() {
            let quantization_matrix = get_quantization_matrix(image.metadata.quality, channel);
            for level in 0..9 {
                for i in (1 << level)..1 << (level + 1) {
                    if let Some(coefficient) = channel_coef.get_mut(i) {
                        if coefficient.is_some() {
                            *coefficient = Some(get_dequantized_value(
                                coefficient.unwrap(),
                                quantization_matrix[level as usize],
                                image.metadata.quality,
                            ))
                        }
                    }
                }
            }
        }
    }

    Ok(image)
}
