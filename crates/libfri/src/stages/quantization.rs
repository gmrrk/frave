use crate::{encoder::EncoderOpts, stages::wavelet_transform::WaveletImage, utils};

fn get_quantization_matrix() -> [i32; 9] {
    return [1;9];
    //return [1, 2, 2, 4, 4, 4, 8, 16, 32];
    //return [1, 2, 3, 3, 4, 5, 7, 8, 7];
    //return [1, 2, 2, 3, 4, 6, 9, 31, 31];
}

pub fn encode(mut image: WaveletImage, encoder_config: &EncoderOpts) -> Result<WaveletImage, String> {
    let quantization_matrix = get_quantization_matrix();//encoder_config.quantization_table;
    for (_, fractal) in &mut image.fractal_lattice {
        for (channel, channel_coef) in fractal.coefficients.iter_mut().enumerate() {
            for level in 0..9 {
                for i in (1<<level)..1<<(level+1) {
                    if let Some(coefficient) = channel_coef.get_mut(i) {
                        if coefficient.is_some() {
                            *coefficient =  Some(coefficient.unwrap() / quantization_matrix[level as usize]);
                        }
                    }
                }
            }
        }
    }

    Ok(image)
}

pub fn decode(mut image: WaveletImage, quantization_matrix: &[i32;9]) -> Result<WaveletImage, String> {
    let quantization_matrixx = get_quantization_matrix();
    for (_, fractal) in &mut image.fractal_lattice {
        for (channel, channel_coef) in fractal.coefficients.iter_mut().enumerate() {
            for level in 0..9 {
                for i in (1<<level)..1<<(level+1) {
                    if let Some(coefficient) = channel_coef.get_mut(i) {
                        if coefficient.is_some() {
                            *coefficient =  Some(coefficient.unwrap() * quantization_matrixx[level as usize]);
                        }
                    }
                }
            }
        }
    }

    Ok(image)
}
