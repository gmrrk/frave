use crate::{complex_plane::ComplexPlane, encoder::EncoderOpts, fractal::{Fractal, BASE_FRAC_DEPTH}};
use crate::images::{ChannelData, CompressedImage};
use crate::stages::prediction;
use crate::stages::wavelet_transform::WaveletImage;
use crate::{complex_plane, utils};

use core::f32;
use num::Complex;
use std::collections::HashMap;
use std::usize;

use rans::b64_decoder::{B64RansDecSymbol, B64RansDecoderMulti};
use rans::b64_encoder::{B64RansEncSymbol, B64RansEncoderMulti};
use rans::RansDecoderMulti;
use rans::RansEncoderMulti;
use rans::{RansDecSymbol, RansEncSymbol};

use crate::stages::prediction::CONTEXT_AMOUNT;

use super::prediction::laplace_distribution;

pub const ALPHABET_SIZE: usize = 1 << 10;

//fn get_first_some_starting_from(i: usize, vec: &Vec<Option<i32>>) -> usize {
//    (i..vec.len()).find(|j| vec[*j].is_some()).unwrap()
//}

#[derive(Debug, Clone)]
pub struct AnsContext {
    pub symbols: Vec<u32>,
    pub freqs: [u32; ALPHABET_SIZE],
    pub cdf: [u32; ALPHABET_SIZE],
    pub freqs_to_enc_symbols: Vec<B64RansEncSymbol>,
    pub freqs_to_dec_symbols: HashMap<u32, B64RansDecSymbol>,
    pub off_distribution_values: Vec<u16>,
    pub max_freq_bits: u32,
    pub width: f32,
}

impl Default for AnsContext {
    fn default() -> Self {
        Self::new()
    }
}

impl AnsContext {
    pub fn new() -> Self {
        AnsContext {
            freqs: [0; ALPHABET_SIZE],
            cdf: [0; ALPHABET_SIZE],
            symbols: (0..ALPHABET_SIZE).map(|x| x as u32).collect(),
            freqs_to_enc_symbols: Vec::new(),
            freqs_to_dec_symbols: HashMap::new(),
            off_distribution_values: Vec::new(),
            max_freq_bits: 8,
            width: 0.0,
        }
    }

    fn get_freqs(coefs: &Vec<u32>) -> [u32; ALPHABET_SIZE] {
        let mut freqs = [0; ALPHABET_SIZE];
        for coef in coefs {
            freqs[*coef as usize] += 1;
        }
        freqs
    }

    fn get_cdf(&self) -> [u32; ALPHABET_SIZE] {
        self.freqs
            .iter()
            .scan(0_u32, |acc, x| {
                let val = *acc;
                *acc += x;
                Some(val)
            })
            .collect::<Vec<u32>>()
            .try_into()
            .unwrap()
    }

    fn update_freqs(&mut self, new_freqs: [u32; ALPHABET_SIZE]) {
        for i in 0..ALPHABET_SIZE {
            self.freqs[i] += new_freqs[i];
        }
    }

    fn fill_with_laplace(&mut self) {
        let width = self.width; 
        for (j, freq) in self.freqs.iter_mut().enumerate() {
            let laplace_value = (laplace_distribution(utils::unpack_signed(j as u32) as f32, 0., width) * (1<<self.max_freq_bits) as f32) as u32;
            if laplace_value == 0 && *freq == 0 && self.off_distribution_values.contains(&(j as u16)) {
                *freq = 1;
            }
            else if *freq != 0 && laplace_value == 0 {
                *freq = 1;
                self.off_distribution_values.push(j as u16);
            } else {
               *freq = laplace_value;
            }
        }
    }

    pub fn bump_freq(&mut self, element: u32) {
        self.freqs[element as usize] += 1;
    }

    fn estimate_width(&mut self) {
        let mut data: Vec<_> = vec![];
        for (j, freq) in self.freqs.iter().enumerate() {
            let sym = utils::unpack_signed(j as u32);
            for _i in 0..*freq {
                data.push(sym.abs());
            }
        }

        self.width = (data.iter().sum::<i32>() as f32 / data.len() as f32).max(1e-3);
    }

    pub fn finalize_context(&mut self, estimate: bool) {
        if self.max_freq_bits < 8 {
            self.max_freq_bits = 8
        }

        if estimate {
            self.estimate_width();
        }
    
        self.fill_with_laplace();

        if self.freqs.iter().sum::<u32>() as usize == 0 {
            return;
        }

        self.cdf = self.normalize_freqs(1 << self.max_freq_bits);
        self.max_freq_bits =
            utils::get_prev_power_two(self.freqs.iter().sum::<u32>() as usize).trailing_zeros();
        self.freqs_to_enc_symbols = self.get_freqs_to_enc_symbols();
        self.freqs_to_dec_symbols = self.get_freqs_to_dec_symbols();
    }

    pub fn normalize_freqs(&mut self, target_total: u32) -> [u32; ALPHABET_SIZE] {
        let mut cum_freqs = self.get_cdf();
        let cur_total = *cum_freqs.last().unwrap() + self.freqs.last().unwrap();
        for i in 1..cum_freqs.len() {
            cum_freqs[i] = ((target_total as u64 * cum_freqs[i] as u64) / cur_total as u64) as u32;
        }

        //Fixing 0 freq values
        for i in 0..cum_freqs.len() - 1 {
            if self.freqs[i] != 0 && cum_freqs[i + 1] == cum_freqs[i] {
                let mut best_freq: u32 = u32::MAX;
                let mut best_steal: usize = usize::MAX;
                for j in 0..cum_freqs.len() - 1 {
                    let freq = cum_freqs[j + 1] - cum_freqs[j];
                    if freq > 1 && freq < best_freq {
                        best_freq = freq;
                        best_steal = j;
                    }
                }
                if best_steal == usize::MAX {
                    continue;
                }

                if best_steal < i {
                    for j in (best_steal + 1)..=i {
                        cum_freqs[j] -= 1;
                    }
                } else {
                    for j in (i + 1)..=best_steal {
                        cum_freqs[j] += 1;
                    }
                }
            }
        }

        for i in 0..(cum_freqs.len() - 1) {
            self.freqs[i] = cum_freqs[i + 1] - cum_freqs[i];
        }
        self.freqs[self.freqs.len() - 1] = cum_freqs[self.freqs.len() - 1] - target_total;
        cum_freqs
    }

    fn get_freqs_to_enc_symbols(&self) -> Vec<B64RansEncSymbol> {
        self.cdf
            .iter()
            .zip(self.freqs.iter())
            .map(|(&cum_freq, &freq)| B64RansEncSymbol::new(cum_freq, freq, self.max_freq_bits))
            .collect()
    }

    fn get_freqs_to_dec_symbols(&self) -> HashMap<u32, B64RansDecSymbol> {
        self.cdf
            .iter()
            .zip(self.freqs.iter())
            .map(|(&cum_freqs, &freqs)| (cum_freqs, B64RansDecSymbol::new(cum_freqs, freqs)))
            .collect()
    }
}

// TODO: Implement Alias sampling method
#[must_use]
fn find_nearest_or_equal(cum_freq: u32, cum_freqs: &[u32]) -> u32 {
    match cum_freqs.binary_search(&cum_freq) {
        Ok(x) => cum_freqs[x],
        Err(x) => cum_freqs[x - 1],
    }
}

pub fn encode_symbol(
    value: i32,
    predicted_value: i32,
    width: usize,
    ans_contexts: &Vec<AnsContext>,
) -> (B64RansEncSymbol, usize) {
    let bucket = width;
    let current_context = &ans_contexts[bucket];
    let symbol_map = &current_context.freqs_to_enc_symbols;
    (
        symbol_map[utils::pack_signed(value - predicted_value) as usize].clone(),
        bucket,
    )
}

fn decode_symbol<const T: usize>(
    image_position: Complex<i32>,
    haar_tree_position: usize,
    depth: usize,
    parent_pos: &Complex<i32>,
    channel: usize,
    ans_contexts: &Vec<AnsContext>,
    complex_plane: &ComplexPlane,
    value_prediction_params: &Vec<[f32; 7]>,
    width_prediction_params: &Vec<[f32; 7]>,
    decoder: &mut B64RansDecoderMulti<T>,
) -> i32 {
    let (bucket, prediction) = if haar_tree_position < 2 {
        prediction::get_lf_context_bucket(
            depth,
            parent_pos,
            complex_plane,
            channel,
        )
    } else {
        prediction::get_hf_context_bucket(
            image_position,
            depth,
            complex_plane,
            value_prediction_params,
            width_prediction_params,
            channel,
        )
    };

    let decoder_pos = CONTEXT_AMOUNT - bucket - 1;
    let current_context = &ans_contexts[bucket];
    let cum_freq_to_symbols = &current_context.freqs_to_dec_symbols;

    let decoded_cdf = decoder.get_at(decoder_pos, current_context.max_freq_bits);
    let cum_freq_decoded = find_nearest_or_equal(decoded_cdf, &current_context.cdf);

    let mut symbol = current_context
        .cdf
        .iter()
        .position(|&r| r == cum_freq_decoded)
        .unwrap() as u32;

    while (symbol as usize) < ALPHABET_SIZE && current_context.cdf[symbol as usize] == cum_freq_decoded  {
        symbol += 1;
    }
    symbol -= 1;

    decoder.advance_step_at(
        decoder_pos,
        &cum_freq_to_symbols[&cum_freq_decoded],
        current_context.max_freq_bits,
    );
    decoder.renorm_at(decoder_pos);
    utils::unpack_signed(symbol) + prediction
}

pub fn encode(
    image: WaveletImage,
    contexts: [Vec<AnsContext>; 3],
    encoder_opts: &EncoderOpts,
) -> Result<CompressedImage, String> {
    let mut channel_data: [Option<ChannelData>; 3] = [None, None, None];
    let sorted_lattice = image.get_sorted_lattice();

    let global_depth = BASE_FRAC_DEPTH;
    for channel in 0..image.metadata.colorspace.num_channels() {
        let mut encoder: B64RansEncoderMulti<CONTEXT_AMOUNT> =
            B64RansEncoderMulti::new(image.complex_plane.fractals.len() * 2 * (1 << global_depth));

        let mut enc_symbols = Vec::<(B64RansEncSymbol, usize)>::new();
        enc_symbols.reserve(1 << global_depth);

        // First scan -> Low frequency coefficients
        for image_pos in sorted_lattice[0].iter() {
            let (fractal, _) = &image.complex_plane.get_parent_fractal_at(*image_pos, 0).unwrap();
            if let Some(value) = fractal.coefficients[channel][0] {
                let (width, prediction) = fractal.parameter_predictors[channel][0];
                let symbol =
                    encode_symbol(value, prediction, width, &contexts[channel]);
                enc_symbols.push(symbol);
            }
        }

        // Second scan -> High frequency coefficient root
        for image_pos in sorted_lattice[0].iter() {
            let (fractal, _) = &image.complex_plane.get_parent_fractal_at(*image_pos, 1).unwrap();
            if let Some(value) = fractal.coefficients[channel][1] {
                let (width, prediction) = fractal.parameter_predictors[channel][1];
                let symbol =
                    encode_symbol(value, prediction, width, &contexts[channel]);
                enc_symbols.push(symbol);
            }
        }

        // Remaining levels
        for level in 0..global_depth-1 {
            for image_pos in sorted_lattice[level].iter() {
                let (fractal, haar_tree_pos) = &image.complex_plane.get_parent_fractal_at(*image_pos, level).unwrap();
                if fractal.coefficients[channel][*haar_tree_pos].is_some() {
                    if let Some(left_coef) = fractal.coefficients[channel][*haar_tree_pos* 2] {
                        let (left_width, left_prediction) = fractal.parameter_predictors[channel][*haar_tree_pos*2];
                        let left_symbol = encode_symbol(
                            left_coef,
                            left_prediction,
                            left_width,
                            &contexts[channel],
                        );
                        enc_symbols.push(left_symbol);
                    }

                    if let Some(right_coef) = fractal.coefficients[channel][*haar_tree_pos* 2 + 1] {
                        let (right_width, right_prediction) = fractal.parameter_predictors[channel][*haar_tree_pos*2+1];

                        let right_symbol = encode_symbol(
                            right_coef,
                            right_prediction,
                            right_width,
                            &contexts[channel],
                        );

                        enc_symbols.push(right_symbol);
                    }
                }
            }
        }

        for (symbol, bucket) in enc_symbols.into_iter().rev() {
            encoder.put_at(bucket, &symbol);
        }
        encoder.flush_all();
        let data = encoder.data().to_owned();
        let bpp = data.len() as f32 / (image.metadata.width * image.metadata.height) as f32 * 8.;
        if true || encoder_opts.verbose {
            println!("bits per pixel: {bpp}");
        }
        channel_data[channel] = Some(ChannelData {
            ans_contexts: contexts[channel].clone(),
            data,
            value_prediction_parameters: encoder_opts.value_prediction_params[channel].clone(),
            width_prediction_parameters: encoder_opts.width_prediction_params[channel].clone(),
        });
    }
    Ok(CompressedImage {
        metadata: image.metadata,
        channel_data,
    })
}

pub fn decode(mut compressed_image: CompressedImage) -> Result<WaveletImage, String> {
    let mut decoded = WaveletImage::from_metadata(compressed_image.metadata);

    let sorted_lattice = decoded.get_sorted_lattice().clone();
    let mut channel = 0;
    let global_depth = BASE_FRAC_DEPTH;
    while let Some(ChannelData {
        ans_contexts,
        data,
        value_prediction_parameters,
        width_prediction_parameters,
    }) = compressed_image.channel_data[channel].take()
    {
        let mut decoder: B64RansDecoderMulti<CONTEXT_AMOUNT> = B64RansDecoderMulti::new(data);
        // First scan -> Low frequency coefficients
        for image_pos in sorted_lattice[0].iter() {
            let symbol = decode_symbol(
                *image_pos,
                0,
                0,
                image_pos,
                channel,
                &ans_contexts,
                &decoded.complex_plane,
                &value_prediction_parameters,
                &width_prediction_parameters,
                &mut decoder,
            );
            let fractal = decoded.complex_plane.get_fractal_mut(*image_pos).unwrap();
            fractal.coefficients[channel][0] = Some(symbol);
        }

        // Second scan -> High frequency coefficient root
        for image_pos in sorted_lattice[0].iter() {
            let symbol = decode_symbol(
                *image_pos,
                1,
                0,
                image_pos,
                channel,
                &ans_contexts,
                &decoded.complex_plane,
                &value_prediction_parameters,
                &width_prediction_parameters,
                &mut decoder,
            );
            let fractal = decoded.complex_plane.get_fractal_mut(*image_pos).unwrap();
            fractal.coefficients[channel][1] = Some(symbol);
        }

        // Remaining levels
        for level in 0..global_depth-1 {
            for image_pos in sorted_lattice[level].iter() {
                let mut left_symbol: Option<i32> = None;
                let mut right_symbol: Option<i32> = None;

                let (fractal, haar_tree_pos) = decoded.complex_plane.get_parent_fractal_at(*image_pos, level).unwrap();
                if fractal.coefficients[channel][haar_tree_pos]
                    .is_none()
                {
                    continue;
                }
                if fractal.coefficients[channel][2*haar_tree_pos]
                    .is_some()
                {
                    left_symbol = Some(decode_symbol(
                        *image_pos,
                        2*haar_tree_pos,
                        level,
                        &fractal.center,
                        channel,
                        &ans_contexts,
                        &decoded.complex_plane,
                        &value_prediction_parameters,
                        &width_prediction_parameters,
                        &mut decoder,
                    ));

                }

                if fractal.coefficients[channel][2*haar_tree_pos+1]
                    .is_some()
                {
                     right_symbol = Some(decode_symbol(
                        *image_pos,
                        2*haar_tree_pos+1,
                        level,
                        &fractal.center,
                        channel,
                        &ans_contexts,
                        &decoded.complex_plane,
                        &value_prediction_parameters,
                        &width_prediction_parameters,
                        &mut decoder,
                    ));

                }

                let fractal_mut = decoded.complex_plane.get_fractal_mut(*image_pos).unwrap();
                fractal_mut.coefficients[channel][2*haar_tree_pos] = left_symbol;
                fractal_mut.coefficients[channel][2*haar_tree_pos+1] = right_symbol;
            }
        }
        channel += 1;
        if channel >= decoded.metadata.colorspace.num_channels() {
            break;
        }
    }

    Ok(decoded)
}

#[cfg(test)]
mod test {
    // #[test]
    // fn cum_sum_test() {
    //     let x = [1, 2, 3, 4, 5];
    //     let cum_x = get_cdf(&x);
    //     assert_eq!(cum_x, &[0, 1, 3, 6, 10])
    // }

    #[test]
    fn logic_test() {}
}
