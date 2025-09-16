use std::array;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

use num::Complex;

use crate::complex_plane::ComplexPlane;
use crate::context_modeling::ContextModeler;
use crate::encoder::EncoderOpts;
use crate::fractal::{Fractal, BASE_FRAC_DEPTH};
use crate::stages::entropy_coding::AnsContext;
use crate::stages::wavelet_transform::WaveletImage;
use crate::{complex_plane, utils};

pub const CONTEXT_AMOUNT: usize = 90;

fn emit_coefficients(data: &[u32], ctx_id: usize, ctx_channel: usize) {
    std::fs::create_dir_all("./coefficients").unwrap();
    let mut f = File::create(format!(
        "coefficients/{ctx_channel}_context_{ctx_id}.coef"
    ))
    .expect("Unable to create coef file");

    for i in data {
        writeln!(f, "{i}").unwrap();
    }
}

fn emit_mse(mse: &Vec<i32>, ctx_channel: usize) {
    std::fs::create_dir_all("./mse").unwrap();
    let mut f = File::create(format!("mse/errors_{ctx_channel}.mse"))
        .expect("Unable to create coef file");
    for i in mse {
        writeln!(f, "{i}").unwrap();
    }
}


pub fn assign_bucket(width: f32, level: usize) -> usize {
    match width as u32 {
        0..3 => level * 10,
        3..5 => level * 10 + 1,
        5..6 => level * 10 + 2,
        6..8 => level * 10 + 3,
        8..12 => level * 10 + 4,
        12..16 => level * 10 + 5,
        16..20 => level * 10 + 6,
        20..25 => level * 10 + 7,
        25..30 => level * 10 + 8,
        30.. => level * 10 + 9,
    }
}

//pub fn get_width_from_bucket(bucket: usize) -> f32 {
//    match bucket % 10 {
//        0 => 2.5,
//        1 => 4.5,
//        2 => 6.3,
//        3 => 8.5,
//        4 => 12.7,
//        5 => 16.,
//        6 => 20.,
//        7 => 24.,
//        8 => 28.,
//        9 => 36.,
//        10.. => 50.
//    }
//}

pub fn get_lf_context_bucket(
    current_depth: usize,
    parent_fractal_pos: &Complex<i32>,
    complex_plane: &ComplexPlane,
    channel: usize,
) -> (usize, i32) {
    let values = ContextModeler::get_neighbour_values(*parent_fractal_pos, current_depth, complex_plane, channel);
    let width: u32 = (values[4] - values[6]).unsigned_abs();

    let bucket = assign_bucket(width as f32, current_depth);

    let prediction = if values[5] >= max(values[4], values[6]) {
        max(values[4], values[6])
    } else if values[5] <= min(values[4], values[6]) {
        min(values[4], values[6])
    } else {
        values[4] + values[6] - values[5]
    };

    (bucket, prediction)
}

pub fn get_hf_context_bucket(
    image_position: Complex<i32>,
    current_depth: usize,
    complex_plane: &ComplexPlane,
    value_prediction_params: &Vec<[f32; 7]>,
    width_prediction_params: &Vec<[f32; 7]>,
    channel: usize,
) -> (usize, i32) {
    let value_prediction_params_layer = value_prediction_params[current_depth+1];
    let width_prediction_params_layer = width_prediction_params[current_depth+1]; 

    let values = ContextModeler::get_neighbour_values(
        image_position,
        current_depth,
        complex_plane,
        channel
    );

    let width = width_prediction_params_layer[0]
        + width_prediction_params_layer[1] * ((values[0] - values[3]).abs() as f32)
        + width_prediction_params_layer[2] * ((values[1] - values[2]).abs() as f32)
        + width_prediction_params_layer[3] * ((values[4] - values[5]).abs() as f32)
        + width_prediction_params_layer[4] * ((values[1] - values[5]).abs() as f32)
        + width_prediction_params_layer[5] * ((values[2] - values[4]).abs() as f32);


    let bucket = assign_bucket(width, current_depth+1);

    let prediction = (values[0] as f32) * value_prediction_params_layer[0]
        + (values[1] as f32) * value_prediction_params_layer[1]
        + (values[2] as f32) * value_prediction_params_layer[2]
        + (values[3] as f32) * value_prediction_params_layer[3]
        + (values[4] as f32) * value_prediction_params_layer[4]
        + (values[5] as f32) * value_prediction_params_layer[5]
        + (values[6] as f32) * value_prediction_params_layer[6];

    (bucket, prediction as i32)
}

fn get_entropy(histogram: &[u32], total_size: usize) -> f32 {
    let mut entropy = 0f32;
    for val in histogram {
        let symbol_prob = *val as f32 / total_size as f32;
        if symbol_prob >= f32::EPSILON {
            entropy += symbol_prob * symbol_prob.log2();
        }
    }
    -entropy
}

pub fn laplace_distribution(x: f32, center: f32, width: f32) -> f32 {
    (-(x-center).abs()/width).exp()/(2.0*width)
}

pub fn encode(
    wavelet_image: &mut WaveletImage,
    encoder_opts: &mut EncoderOpts,
) -> Result<[Vec<AnsContext>; 3], String> {
    let mut contexts: [Vec<AnsContext>; 3] = [vec![], vec![], vec![]];
    let sorted_lattice = wavelet_image.get_sorted_lattice().clone();
    for channel in 0..wavelet_image.metadata.colorspace.num_channels() {
        contexts[channel] = vec![AnsContext::new(); CONTEXT_AMOUNT];
        let mut mse: Vec<i32> = vec![];
        let depth = BASE_FRAC_DEPTH;

        for image_pos in sorted_lattice[0].iter() {
            let (fractal, _) = &wavelet_image.complex_plane.get_parent_fractal_at(*image_pos, 0).unwrap();
            if let Some(value) = fractal.coefficients[channel][0] {
                let (bucket, prediction) =
                    get_lf_context_bucket(0, image_pos, &wavelet_image.complex_plane, channel);
                let residual = value - prediction;
                {
                    let mut_frac = wavelet_image.complex_plane.get_fractal_mut(*image_pos).unwrap();
                    mut_frac.parameter_predictors[channel][0] = (bucket, prediction);
                    contexts[channel][bucket].bump_freq(utils::pack_signed(residual));
                }
            }
        }

        // Second scan -> High frequency coefficient root
        for image_pos in sorted_lattice[0].iter() {
            let (fractal, _) = &wavelet_image.complex_plane.get_parent_fractal_at(*image_pos, 1).unwrap();
            if let Some(value) = fractal.coefficients[channel][1] {
                let (bucket, prediction) =
                    get_lf_context_bucket(1, image_pos, &wavelet_image.complex_plane, channel);

                let residual = value - prediction;
                {
                    let mut_frac = wavelet_image.complex_plane.get_fractal_mut(*image_pos).unwrap();
                    mut_frac.parameter_predictors[channel][1] = (bucket, prediction);
                    contexts[channel][bucket].bump_freq(utils::pack_signed(residual));
                }
            }

        }

        for level in 0..depth-1 {
            for image_pos in sorted_lattice[level].iter() {
                let (fractal, haar_tree_pos) = wavelet_image.complex_plane.get_parent_fractal_at(*image_pos, level).unwrap();
                if fractal.coefficients[channel][haar_tree_pos].is_some() {
                    let (bucket, prediction) = get_hf_context_bucket(
                        *image_pos,
                        level,
                        &wavelet_image.complex_plane,
                        &encoder_opts.value_prediction_params[channel],
                        &encoder_opts.width_prediction_params[channel],
                        channel,
                    );
    
                    let residual_left = if let Some(left_coef) = fractal.coefficients[channel][2*haar_tree_pos] {
                        left_coef - prediction
                    } else {
                        0
                    };

                    let residual_right = if let Some(right_coef) = fractal.coefficients[channel][2*haar_tree_pos+1] {
                        right_coef - prediction
                    } else {
                        0
                    };

                    mse.push((residual_left).pow(2));
                    mse.push((residual_right).pow(2));
                    contexts[channel][bucket].bump_freq(utils::pack_signed(residual_left));
                    contexts[channel][bucket].bump_freq(utils::pack_signed(residual_right));

                    let mut_frac = wavelet_image.complex_plane.get_fractal_mut(*image_pos).unwrap();
                    mut_frac.parameter_predictors[channel][2*haar_tree_pos] = (bucket, prediction);
                    mut_frac.parameter_predictors[channel][2*haar_tree_pos+1] = (bucket, prediction);
                }
            }
        }

        //emit_mse(&mse, channel);

        for (i, ctx) in contexts[channel].iter_mut().enumerate() {
            ctx.max_freq_bits =
                utils::get_prev_power_two(ctx.freqs.iter().sum::<u32>() as usize).trailing_zeros();
            ctx.finalize_context(true);
            if encoder_opts.verbose {
                println!(
                    "CHANNEL: {}, size: {}, entropy: {}",
                    channel,
                    ctx.freqs.iter().sum::<u32>() as usize,
                    get_entropy(&ctx.freqs, ctx.freqs.iter().sum::<u32>() as usize)
                );
            }

            if encoder_opts.emit_coefficients {
                emit_coefficients(&ctx.freqs, i, channel)
            }
        }
    }

    Ok(contexts)
}
