use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::vec;

use crate::context_modeling::ContextModeler;
use crate::encoder::EncoderOpts;
use crate::fractal::{self, Fractal, BASE_FRAC_DEPTH, LITERALS};
use crate::images::{ImageMetadata, RasterImage};
use crate::utils;

use itertools::Position;
use num::complex::ComplexFloat;
use num::{Complex, Float};

fn try_apply<T: Copy>(
    first: Option<T>,
    second: Option<T>,
    operation: fn(T, T) -> T,
    default: T,
) -> Option<T> {
    match (first, second) {
        (Some(f), Some(s)) => Some(operation(f, s)),
        (Some(f), None) => Some(operation(f, default)),
        (None, Some(s)) => Some(operation(default, s)),
        (None, None) => None,
    }
}

impl RasterImage {
    pub fn from_wavelet(wavelet_image: WaveletImage) -> RasterImage {
        let mut raster = RasterImage {
            data: vec![
                0;
                wavelet_image.metadata.height as usize
                    * wavelet_image.metadata.width as usize
                    * wavelet_image.metadata.colorspace.num_channels()
            ],
            metadata: wavelet_image.metadata,
        };

        for (_center, fractal) in wavelet_image.fractal_lattice.iter() {
            raster.extract_values(&fractal);
        }

        return raster;
    }

    fn extract_values(&mut self, fractal: &Fractal) {
        for channel in 0..self.metadata.colorspace.num_channels() {
            let mut low_pass_values = vec![0; 1 << fractal.depth];
            low_pass_values[1] = fractal.coefficients[channel][0].unwrap();

            for level in 0..fractal.depth {
                for pos in 1 << level..1 << (level + 1) {
                    if let Some(dif) = fractal.coefficients[channel][pos] {
                        let right_subtree: i32 = low_pass_values[pos] - dif / 2;
                        let left_subtree: i32 = dif + right_subtree;
                        if level == fractal.depth - 1 {
                            let left_pixel = fractal.image_positions[2 * pos];
                            let right_pixel = fractal.image_positions[2 * pos + 1];
                            self.set_pixel(left_pixel.re, left_pixel.im, left_subtree, channel);
                            self.set_pixel(right_pixel.re, right_pixel.im, right_subtree, channel);
                        } else {
                            low_pass_values[2 * pos] = left_subtree;
                            low_pass_values[2 * pos + 1] = right_subtree;
                        }
                    }
                }
            }
        }
    }
}

impl Fractal {
    pub fn extract_coefficients(&mut self, raster_image: &RasterImage, depth: usize) {
        let mut coefficients = [
            vec![None; 1 << depth],
            vec![None; 1 << depth],
            vec![None; 1 << depth],
        ];

        let mut low_pass_values = [
            vec![None; 1 << depth],
            vec![None; 1 << depth],
            vec![None; 1 << depth],
        ];
        for channel in 0..raster_image.metadata.colorspace.num_channels() {
            for level in (0..depth).rev() {
                // compute high-pass and low-pass components
                for pos in 1 << level..1 << (level + 1) {
                    let (left_coef, right_coef): (Option<i32>, Option<i32>);
                    if level == depth - 1 {
                        left_coef = raster_image.get_pixel(
                            self.image_positions[2 * pos].re,
                            self.image_positions[2 * pos].im,
                            channel,
                        );
                        right_coef = raster_image.get_pixel(
                            self.image_positions[2 * pos + 1].re,
                            self.image_positions[2 * pos + 1].im,
                            channel,
                        );
                    } else {
                        left_coef = low_pass_values[channel][2 * pos];
                        right_coef = low_pass_values[channel][2 * pos + 1];
                    }
                    coefficients[channel][pos] =
                        try_apply(left_coef, right_coef, |l, r| (l - r), 0);
                    low_pass_values[channel][pos] = try_apply(
                        right_coef,
                        coefficients[channel][pos],
                        |l, r| (l + r / 2),
                        0,
                    );
                }
            }
            coefficients[channel][0] = low_pass_values[channel][1];
        }
        self.values = low_pass_values;
        self.coefficients = coefficients;
    }
}

pub struct WaveletImage {
    pub metadata: ImageMetadata,
    pub fractal_lattice: HashMap<Complex<i32>, Fractal>,
    pub global_position_map: Vec<HashMap<Complex<i32>, Complex<i32>>>,
    pub sorted_lattice: [Vec<Complex<i32>>; BASE_FRAC_DEPTH as usize],
}

impl WaveletImage {
    pub fn from_metadata(metadata: ImageMetadata) -> WaveletImage {
        let image = RasterImage {
            data: vec![
                0;
                (metadata.width) as usize
                    * (metadata.height) as usize
                    * metadata.colorspace.num_channels()
            ],
            metadata,
        };
        return Self::from_raster(image);
    }

    pub fn from_raster(raster_image: RasterImage) -> WaveletImage {
        let mut fractal_lattice = Self::fractal_divide(
            raster_image.metadata.width,
            raster_image.metadata.height,
            BASE_FRAC_DEPTH,
        );

        for (_, fractal) in fractal_lattice.iter_mut() {
            fractal.extract_coefficients(&raster_image, fractal.depth);
        }
        fractal_lattice
            .retain(|_, frac| frac.coefficients.iter().any(|channel| channel[0].is_some()));

        let global_position_map = Self::get_global_position_map(&fractal_lattice);
        let sorted_lattice = Self::sort_lattice(
            &fractal_lattice,
            &global_position_map,
            raster_image.metadata.height,
            raster_image.metadata.width,
        );

        WaveletImage {
            metadata: raster_image.metadata,
            global_position_map,
            fractal_lattice,
            sorted_lattice,
        }
    }

    fn get_global_position_map(
        fractal_lattice: &HashMap<Complex<i32>, Fractal>,
    ) -> Vec<HashMap<Complex<i32>, Complex<i32>>> {
        let mut position_map = vec![HashMap::new(); BASE_FRAC_DEPTH as usize];

        for (center, frac) in fractal_lattice.iter() {
            for level in 0..BASE_FRAC_DEPTH {
                for position in &frac.image_positions[1 << level..(1 << level + 1)] {
                    position_map[level as usize].insert(*position, *center);
                }
            }
        }

        position_map
    }

    fn fractal_divide(width: u32, height: u32, depth: usize) -> HashMap<Complex<i32>, Fractal> {
        let mut fractal_lattice = HashMap::<Complex<i32>, Fractal>::new();
        let center = Complex::<i32>::new(width as i32 / 2, height as i32 / 2);
        let mut to_add = VecDeque::<Complex<i32>>::new();
        to_add.push_back(center);

        let mut boundary = VecDeque::<Complex<i32>>::new();

        while let Some(position) = to_add.pop_front() {
            if position.re < 0
                || position.im < 0
                || position.re > width as i32
                || position.im > height as i32
            {
                boundary.push_back(position);
                continue;
            }

            let fractal = Fractal::new(depth, position);
            for neighbour in fractal.get_neighbour_locations() {
                if !fractal_lattice.contains_key(&neighbour) && !to_add.contains(&neighbour) {
                    to_add.push_back(neighbour);
                }
            }

            fractal_lattice.insert(position, fractal);
        }

        while let Some(position) = boundary.pop_front() {
            let boundary_fractal = Fractal::new(depth, position);
            fractal_lattice.insert(position, boundary_fractal);
        }

        fractal_lattice
    }

    pub fn get_sorted_lattice(&self) -> &[Vec<Complex<i32>>; BASE_FRAC_DEPTH as usize] {
        &self.sorted_lattice
    }

    fn is_pos_in_row_boundary(
        pos: &Complex<i32>,
        row_dir: &Complex<i32>,
        min_real: i32,
        max_real: i32,
        min_imag: i32,
        max_imag: i32,
    ) -> bool {
        if row_dir.re.abs() > row_dir.im.abs() {
            pos.im >= min_imag && pos.im <= max_imag
        } else {
            pos.re >= min_real && pos.re <= max_real
        }
    }

    fn scan_level(
        level: usize,
        depth: usize,
        center: Complex<i32>,
        global_position_map: &HashMap<Complex<i32>, Complex<i32>>,
        min_real: i32,
        max_real: i32,
        min_imag: i32,
        max_imag: i32,
    ) -> Vec<Complex<i32>> {
        let neighbour_vectors = Fractal::get_nearby_vectors(BASE_FRAC_DEPTH - level);
        let row_dir = neighbour_vectors[3];
        let rev_row_dir = neighbour_vectors[0];

        let col_dir = neighbour_vectors[1];
        let rev_col_dir = neighbour_vectors[4];
        //find first

        let mut first = center;

        let mut layer_seven_mod = 0;
        if !global_position_map.contains_key(&(center + rev_row_dir))
            && global_position_map.contains_key(&(center + Complex::new(-1, -1)))
        {
            layer_seven_mod = 1;
        }
        let mut last_seen = first;

        while (global_position_map.contains_key(&first)) {
            last_seen = first;
            if depth - level != 2 {
                first += rev_row_dir;
            } else {
                if layer_seven_mod % 2 == 0 {
                    first += rev_row_dir
                } else {
                    first += Complex::new(-1, -1);
                }
                layer_seven_mod += 1;
            }
        }

        // Find first row
        loop {
            let mut column_forward = first;
            let mut column_backward = first;
            let mut empty_column = true;
            while (column_forward.im <= max_imag && column_forward.im >= min_imag)
                || (column_backward.im <= max_imag && column_backward.im >= min_imag)
                || (column_forward.re <= max_real && column_forward.re >= min_real)
                || (column_backward.re <= max_real && column_backward.re >= min_real)
            {
                column_forward += col_dir;
                column_backward += rev_col_dir;
                if global_position_map.contains_key(&column_forward) {
                    last_seen = column_forward;
                    empty_column = false;
                    break;
                }
                if global_position_map.contains_key(&column_backward) {
                    last_seen = column_backward;
                    empty_column = false;
                    break;
                }
            }
            if empty_column {
                first = last_seen;
                break;
            } else {
                if depth - level != 2 {
                    first += rev_row_dir;
                } else {
                    if layer_seven_mod % 2 == 0 {
                        first += rev_row_dir
                    } else {
                        first += Complex::new(-1, -1);
                    }
                    layer_seven_mod += 1;
                }
            }
        }

        // Scanning backwards find first column
        while (first.im <= max_imag
            && first.im >= min_imag
            && first.re <= max_real
            && first.re >= min_real)
        {
            first += rev_col_dir;
            if global_position_map.contains_key(&first) {
                last_seen = first;
            }
        }
        first = last_seen;
        layer_seven_mod = 1;

        // Fill plane in sorted order
        let mut plane: Vec<Complex<i32>> = Vec::new();
        'outer: loop {
            let mut cnt = 0;
            let mut scan = first;
            loop {
                if global_position_map.contains_key(&scan) {
                    plane.push(scan);
                    cnt += 1;
                }
                if ((scan.im > max_imag || scan.im < min_imag)
                    || (col_dir.im == 0 && (scan.re > max_real || scan.re < min_real)))
                {
                    break;
                }
                scan += col_dir;
            }

            if depth - level != 2 {
                first += row_dir;
            } else {
                if layer_seven_mod % 2 == 0 {
                    first += Complex::new(1, 1);
                } else {
                    first += row_dir
                }
                layer_seven_mod += 1;
            }
            while (!global_position_map.contains_key(&first)) {
                first += col_dir;
                if (!Self::is_pos_in_row_boundary(
                    &first, &row_dir, min_real, max_real, min_imag, max_imag,
                )) {
                    break 'outer;
                }
            }
            if global_position_map.contains_key(&first) {
                last_seen = first;
                while (first.im <= max_imag
                    && first.im >= min_imag
                    && first.re <= max_real
                    && first.re >= min_real)
                {
                    first += rev_col_dir;

                    if global_position_map.contains_key(&first) {
                        last_seen = first;
                    }
                }
                first = last_seen;
            }
        }
        plane
    }

    // TODO: Simplify this logic from hell
    fn sort_lattice(
        fractal_lattice: &HashMap<Complex<i32>, Fractal>,
        global_position_map: &Vec<HashMap<Complex<i32>, Complex<i32>>>,
        height: u32,
        width: u32,
    ) -> [Vec<Complex<i32>>; BASE_FRAC_DEPTH as usize] {
        let keys: Vec<Complex<i32>> = fractal_lattice.keys().cloned().collect();
        let depth = fractal_lattice[&keys[0]].depth;

        let min_real = global_position_map[BASE_FRAC_DEPTH as usize - 1]
            .keys()
            .min_by_key(|x| x.re)
            .unwrap()
            .re;
        let max_real = global_position_map[BASE_FRAC_DEPTH as usize - 1]
            .keys()
            .max_by_key(|x| x.re)
            .unwrap()
            .re;
        let min_imag = global_position_map[BASE_FRAC_DEPTH as usize - 1]
            .keys()
            .min_by_key(|x| x.im)
            .unwrap()
            .im;
        let max_imag = global_position_map[BASE_FRAC_DEPTH as usize - 1]
            .keys()
            .max_by_key(|x| x.im)
            .unwrap()
            .im;

        let mut sorted_fractalwise: [Vec<Complex<i32>>; BASE_FRAC_DEPTH as usize] = Default::default();
        let center = Complex::<i32>::new(width as i32 / 2, height as i32 / 2);

        for level in (0..BASE_FRAC_DEPTH) {
            let plane = Self::scan_level(
                level,
                depth,
                center,
                &global_position_map[level as usize],
                min_real,
                max_real,
                min_imag,
                max_imag,
            );
            assert_eq!(plane.len(), fractal_lattice.len() * (1 << level));
            sorted_fractalwise[level as usize] = plane;
        }
        sorted_fractalwise
    }
}

pub fn encode(
    raster_image: RasterImage,
    encoder_opts: &mut EncoderOpts,
) -> Result<WaveletImage, String> {
    let wavelet_image = WaveletImage::from_raster(raster_image);
    let mut ctx_mod = ContextModeler::new();
    let sorted_lattice = wavelet_image.get_sorted_lattice().clone();
    for channel in 0..wavelet_image.metadata.colorspace.num_channels() {
        ctx_mod.optimize_parameters(&wavelet_image, channel);

        encoder_opts.value_prediction_params[channel] = ctx_mod.value_predictors[channel].clone();
        encoder_opts.width_prediction_params[channel] = ctx_mod.width_predictors[channel].clone();
    }
    Ok(wavelet_image)
}

pub fn decode(wavelet_image: WaveletImage) -> Result<RasterImage, String> {
    Ok(RasterImage::from_wavelet(wavelet_image))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn extract_coefficient_test() {
        //let img = RasterImage {
        //    metadata: ImageMetadata {
        //        height: 8,
        //        width: 8,
        //        colorspace: crate::images::ColorSpace::RGB,
        //        variant: crate::images::FractalVariant::TameTwindragon,
        //    },
        //    data: vec![10; 8 * 8 * 3],
        //};

        //let (depth, center) = calculate_depth_center(img.metadata.width, img.metadata.height);

        //let coefficients = extract_coefficients(&img, center, depth - 1);
    }
}
