use std::collections::{HashMap, VecDeque};
use std::vec;
use std::array;

use crate::context_modeling::ContextModeler;
use crate::encoder::EncoderOpts;
use crate::fractal::{Fractal, BASE_FRAC_DEPTH};
use crate::images::{ImageMetadata, RasterImage};
use crate::complex_plane::ComplexPlane;

use num::Complex;

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

        for fractal in wavelet_image.complex_plane.fractals.iter() {
            raster.extract_values(fractal);
        }

        raster
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
        self.coefficients = coefficients;
    }
}

pub struct WaveletImage {
    pub metadata: ImageMetadata,
    pub sorted_lattice: [Vec<Complex<i32>>; BASE_FRAC_DEPTH],
    pub complex_plane: ComplexPlane,
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
        Self::from_raster(image)
    }

    pub fn from_raster(raster_image: RasterImage) -> WaveletImage {
        let mut fractals = Self::fractal_divide(
            raster_image.metadata.width,
            raster_image.metadata.height,
            BASE_FRAC_DEPTH,
        );

        for fractal in fractals.iter_mut() {
            fractal.extract_coefficients(&raster_image, fractal.depth);
        }
        fractals
            .retain(|frac| frac.coefficients.iter().any(|channel| channel[0].is_some()));

        let plane = ComplexPlane::new(fractals);

        let sorted_lattice = Self::sort_lattice(
            &plane,
            raster_image.metadata.height,
            raster_image.metadata.width,
        );

        WaveletImage {
            complex_plane: plane,
            metadata: raster_image.metadata,
            sorted_lattice,
        }
    }

    fn fractal_divide(width: u32, height: u32, depth: usize) -> Vec<Fractal> {
        let mut fractals = HashMap::<Complex<i32>, Fractal>::new();
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
                if !fractals.contains_key(&neighbour) && !to_add.contains(&neighbour) {
                    to_add.push_back(neighbour);
                }
            }

            fractals.insert(position, fractal);
        }

        while let Some(position) = boundary.pop_front() {
            let boundary_fractal = Fractal::new(depth, position);
            fractals.insert(position, boundary_fractal);
        }

        fractals.into_values().collect()
    }

    pub fn get_sorted_lattice(&self) -> &[Vec<Complex<i32>>; BASE_FRAC_DEPTH] {
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
        center: Complex<i32>,
        complex_cell_plane: &ComplexPlane,
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
        if BASE_FRAC_DEPTH - level == 2
            && !complex_cell_plane.is_cell_on_level(center + rev_row_dir, level)
            && complex_cell_plane.is_cell_on_level(center + Complex::new(-1, -1), level)
        {
            layer_seven_mod = 1;
        }
        let mut last_seen = first;

        while complex_cell_plane.is_cell_on_level(first, level) {
            last_seen = first;
            if BASE_FRAC_DEPTH - level != 2 {
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
                if complex_cell_plane.is_cell_on_level(column_forward, level) {
                    last_seen = column_forward;
                    empty_column = false;
                    break;
                }
                if complex_cell_plane.is_cell_on_level(column_backward, level) {
                    last_seen = column_backward;
                    empty_column = false;
                    break;
                }
            }
            if empty_column {
                first = last_seen;
                break;
            } else if BASE_FRAC_DEPTH - level != 2 {
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

        // Scanning backwards find first column
        while first.im <= max_imag
            && first.im >= min_imag
            && first.re <= max_real
            && first.re >= min_real
        {
            first += rev_col_dir;
            if complex_cell_plane.is_cell_on_level(first, level) {
                last_seen = first;
            }
        }
        first = last_seen;
        layer_seven_mod = 1;

        // Fill plane in sorted order
        let mut plane: Vec<Complex<i32>> = Vec::new();
        'outer: loop {
            let mut scan = first;
            loop {
                if complex_cell_plane.is_cell_on_level(scan, level) {
                    plane.push(scan);
                }
                if (scan.im > max_imag || scan.im < min_imag)
                    || (col_dir.im == 0 && (scan.re > max_real || scan.re < min_real))
                {
                    break;
                }
                scan += col_dir;
            }

            if BASE_FRAC_DEPTH - level != 2 {
                first += row_dir;
            } else {
                if layer_seven_mod % 2 == 0 {
                    first += Complex::new(1, 1);
                } else {
                    first += row_dir
                }
                layer_seven_mod += 1;
            }
            while !complex_cell_plane.is_cell_on_level(first, level) {
                first += col_dir;
                if !Self::is_pos_in_row_boundary(
                    &first, &row_dir, min_real, max_real, min_imag, max_imag,
                ) {
                    break 'outer;
                }
            }
            if complex_cell_plane.is_cell_on_level(first, level) {
                last_seen = first;
                while first.im <= max_imag
                    && first.im >= min_imag
                    && first.re <= max_real
                    && first.re >= min_real
                {
                    first += rev_col_dir;

                    if complex_cell_plane.is_cell_on_level(first, level) {
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
        complex_plane: &ComplexPlane,
        height: u32,
        width: u32,
    ) -> [Vec<Complex<i32>>; BASE_FRAC_DEPTH] {
        let (min_real, min_imag, max_real, max_imag) = complex_plane.get_minmax_coordinates();

        let mut sorted_fractalwise: [Vec<Complex<i32>>; BASE_FRAC_DEPTH] = Default::default();
        let center = Complex::<i32>::new(width as i32 / 2, height as i32 / 2);

        for level in 0..BASE_FRAC_DEPTH {
            let plane_order = Self::scan_level(
                level,
                center,
                complex_plane,
                min_real,
                max_real,
                min_imag,
                max_imag,
            );
            assert_eq!(plane_order.len(), complex_plane.fractals.len() * (1 << level));
            sorted_fractalwise[level] = plane_order;
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
