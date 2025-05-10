use std::cmp::{max, min};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

use num::pow::Pow;
use num::{Complex, PrimInt};

use crate::context_modeling::ContextModeler;
use crate::encoder::EncoderOpts;
use crate::stages::entropy_coding::AnsContext;
use crate::stages::wavelet_transform::{Fractal, WaveletImage};
use crate::{fractal, utils};

pub const CONTEXT_AMOUNT: usize = 10;

fn emit_coefficients(data: &[u32], ctx_id: usize, ctx_channel: usize) {
    std::fs::create_dir_all("./coefficients").unwrap();
    let mut f = File::create(format!(
        "coefficients/{}_context_{}.coef",
        ctx_channel, ctx_id
    ))
    .expect("Unable to create coef file");

    for i in data {
        write!(f, "{}\n", i).unwrap();
    }
}

fn emit_mse(mse: &Vec<i32>, ctx_channel: usize) {
    std::fs::create_dir_all("./mse").unwrap();
    let mut f = File::create(format!("mse/errors_{}.mse", ctx_channel))
        .expect("Unable to create coef file");
    for i in mse {
        write!(f, "{}\n", i).unwrap();
    }
}

fn get_containing_fractal(
    pos: &Complex<i32>,
    level: usize,
    fractal: &Fractal,
    fractal_lattice: &HashMap<Complex<i32>, Fractal>,
) -> Option<Complex<i32>> {
    for location in fractal.get_neighbour_locations() {
        if let Some(neighbour) = fractal_lattice.get(&location) {
            if neighbour.position_map[level].contains_key(&pos) {
                return Some(location);
            }
        }
    }
    None
}

pub fn assign_bucket(width: f32) -> usize {
    match width as u32 {
        0..3 => 0,
        3..5 => 1,
        5..6 => 2,
        6..8 => 3,
        8..12 => 4,
        12..16 => 5,
        16..20 => 6,
        20..25 => 7,
        25..30 => 8,
        30.. => 9,
    }
}

pub fn get_width_from_bucket(bucket: usize) -> f32 {
    match bucket as u32 {
        0 => 2.5,
        1 => 4.5,
        2 => 6.3,
        3 => 8.5,
        4 => 12.7,
        5 => 16.,
        6 => 20.,
        7 => 24.,
        8 => 28.,
        9 => 36.,
        10.. => 50.
    }
}

pub fn get_lf_context_bucket(
    position: usize,
    current_depth: u8,
    parent_fractal_pos: &Complex<i32>,
    fractal_lattice: &HashMap<Complex<i32>, Fractal>,
    channel: usize,
) -> (usize, i32) {
    let fractal = &fractal_lattice[parent_fractal_pos];
    let position_in_image = fractal.image_positions[position];
    let global_pos = vec![];
    let neighbours = vec![
        Fractal::get_left(
            position_in_image,
            fractal.depth - current_depth,
            &global_pos,
        ),
        Fractal::get_up_left(
            position_in_image,
            fractal.depth - current_depth,
            &global_pos,
        ),
        Fractal::get_up_right(
            position_in_image,
            fractal.depth - current_depth,
            &global_pos,
        ),
    ];
    let level: usize = current_depth as usize;

    let values: Vec<i32> = neighbours
        .iter()
        .map(|pos| {
            if fractal.position_map[level].get(pos).is_none() {
                if let Some(nposition) =
                    get_containing_fractal(pos, level, fractal, fractal_lattice)
                {
                    let containing_fractal = &fractal_lattice[&nposition];
                    containing_fractal.coefficients[channel][position].unwrap_or(0)
                } else {
                    0
                }
            } else {
                let loc = fractal.position_map[level][pos];
                fractal.coefficients[channel][loc].unwrap_or(0)
            }
        })
        .collect();

    let width: u32 = (values[0] - values[2]).abs() as u32;

    let bucket = assign_bucket(width as f32);

    let prediction = if values[1] >= max(values[0], values[2]) {
        max(values[0], values[2])
    } else if values[1] <= min(values[0], values[2]) {
        min(values[0], values[2])
    } else {
        values[0] + values[2] - values[1]
    };
    //let bucket = 0;
    //let prediction = 0;

    (bucket, prediction as i32)
}

pub fn get_hf_context_bucket(
    image_position: Complex<i32>,
    current_depth: u8,
    parent_fractal_pos: &Complex<i32>,
    fractal_lattice: &HashMap<Complex<i32>, Fractal>,
    global_position_map: &Vec<HashMap<Complex<i32>, Complex<i32>>>,
    value_prediction_params: &Vec<[f32; 6]>,
    width_prediction_params: &Vec<[f32; 6]>,
    channel: usize,
) -> (usize, i32) {
    assert!(current_depth > 0);

    let depth = fractal_lattice[parent_fractal_pos].depth;

    let value_prediction_params_layer = if current_depth < depth - 2 {
        value_prediction_params[2]
    } else if current_depth == depth - 2 {
        value_prediction_params[1]
    } else {
        value_prediction_params[0]
    };

    let width_prediction_params_layer = if current_depth < depth - 2 {
        width_prediction_params[2]
    } else if current_depth == depth - 2 {
        width_prediction_params[1]
    } else {
        width_prediction_params[0]
    };

    let values = ContextModeler::get_neighbour_values(
        image_position,
        current_depth,
        parent_fractal_pos,
        fractal_lattice,
        global_position_map,
        channel,
    );

    let width = width_prediction_params_layer[0]
        + width_prediction_params_layer[1] * ((values[0] - values[3]).abs() as f32)
        + width_prediction_params_layer[2] * ((values[1] - values[2]).abs() as f32)
        + width_prediction_params_layer[3] * ((values[4] - values[5]).abs() as f32)
        + width_prediction_params_layer[4] * ((values[1] - values[5]).abs() as f32)
        + width_prediction_params_layer[5] * ((values[2] - values[4]).abs() as f32);

    let bucket = assign_bucket(width);

    let prediction = (values[0] as f32) * value_prediction_params_layer[0]
        + (values[1] as f32) * value_prediction_params_layer[1]
        + (values[2] as f32) * value_prediction_params_layer[2]
        + (values[3] as f32) * value_prediction_params_layer[3]
        + (values[4] as f32) * value_prediction_params_layer[4]
        + (values[5] as f32) * value_prediction_params_layer[5];

    (bucket as usize, prediction as i32)
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
    let mut ctx_mod = ContextModeler::new();
    let sorted_lattice = wavelet_image.get_sorted_lattice().clone();
    for channel in 0..wavelet_image.metadata.colorspace.num_channels() {
        ctx_mod.optimize_parameters(&wavelet_image, channel);

        encoder_opts.value_prediction_params[channel] = ctx_mod.value_predictors[channel].clone();
        encoder_opts.width_prediction_params[channel] = ctx_mod.width_predictors[channel].clone();

        contexts[channel] = vec![AnsContext::new(); CONTEXT_AMOUNT];
        let mut mse: Vec<i32> = vec![];
        let depth = wavelet_image.fractal_lattice[&sorted_lattice[0][0]].depth;

        for (i, image_pos) in sorted_lattice[0].iter().enumerate() {
            let fractal = &wavelet_image.fractal_lattice.get(image_pos).unwrap();
            let haar_tree_pos = fractal.position_map[0].get(&image_pos).unwrap();
            if let Some(value) = fractal.coefficients[channel][0] {
                let (bucket, prediction) =
                    get_lf_context_bucket(0, 0, image_pos, &wavelet_image.fractal_lattice, channel);
                let residual = value - prediction;
                {
                    let mut mut_frac = wavelet_image.fractal_lattice.get_mut(image_pos).unwrap();
                    mut_frac.parameter_predictors[channel][0] = (bucket, prediction);
                    contexts[channel][bucket].bump_freq(utils::pack_signed(residual));
                }
            }
        }

        // Second scan -> High frequency coefficient root
        for (i, image_pos) in sorted_lattice[0].iter().enumerate() {
            let fractal = &wavelet_image.fractal_lattice.get(image_pos).unwrap();
            let haar_tree_pos = fractal.position_map[0].get(&image_pos).unwrap();
            if let Some(value) = fractal.coefficients[channel][1] {
                let (bucket, prediction) =
                    get_lf_context_bucket(1, 0, image_pos, &wavelet_image.fractal_lattice, channel);
                let residual = value - prediction;
                {
                    let mut mut_frac = wavelet_image.fractal_lattice.get_mut(image_pos).unwrap();
                    mut_frac.parameter_predictors[channel][1] = (bucket, prediction);
                    contexts[channel][bucket].bump_freq(utils::pack_signed(residual));
                }
            }
        }

        for level in (1..depth).rev() {
            for (i, image_pos) in sorted_lattice[level as usize].iter().enumerate() {
                let parent_pos = wavelet_image.global_position_map[level as usize][&image_pos];
                let fractal = &wavelet_image.fractal_lattice.get(&parent_pos).unwrap();
                let haar_tree_pos = fractal.position_map[level as usize]
                    .get(&image_pos)
                    .unwrap()
                    .clone();
                if let Some(value) = fractal.coefficients[channel][haar_tree_pos] {
                    let (bucket, prediction) = get_hf_context_bucket(
                        *image_pos,
                        level,
                        &parent_pos,
                        &wavelet_image.fractal_lattice,
                        &wavelet_image.global_position_map,
                        &encoder_opts.value_prediction_params[channel],
                        &encoder_opts.width_prediction_params[channel],
                        channel,
                    );
                    let residual = value - prediction;
                    mse.push((residual).pow(2));
                    contexts[channel][bucket].bump_freq(utils::pack_signed(residual));
                    let mut mut_frac = wavelet_image.fractal_lattice.get_mut(&parent_pos).unwrap();
                    mut_frac.parameter_predictors[channel][haar_tree_pos] = (bucket, prediction);
                }
            }
        }

        emit_mse(&mse, channel);

        for (i, ctx) in contexts[channel].iter_mut().enumerate() {
            ctx.max_freq_bits =
                utils::get_prev_power_two(ctx.freqs.iter().sum::<u32>() as usize).trailing_zeros();
            ctx.finalize_context(true, i);
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
    //dbg!(&ctx_mod);

    Ok(contexts)
}
