use std::collections::HashMap;

use lstsq::lstsq;
use nalgebra::Dynamic;
use nalgebra::{self as na, DMatrix, DVector, U2};
use num::Complex;

use crate::fractal::{Fractal, BASE_FRAC_DEPTH};
use crate::stages::wavelet_transform::WaveletImage;

#[derive(Debug)]
pub struct ContextModeler {
    pub value_predictors: [Vec<[f32; 7]>; BASE_FRAC_DEPTH],
    pub width_predictors: [Vec<[f32; 7]>; BASE_FRAC_DEPTH],
}

impl ContextModeler {
    pub fn new() -> Self {
        ContextModeler {
            value_predictors: Default::default(),
            width_predictors: Default::default(),
        }
    }

    pub fn get_neighbour_values(
        image_position: Complex<i32>,
        current_depth: usize,
        parent_fractal_pos: &Complex<i32>,
        fractal_lattice: &HashMap<Complex<i32>, Fractal>,
        global_position_map: &Vec<HashMap<Complex<i32>, Complex<i32>>>,
        channel: usize,
        multidimensional: bool,
    ) -> Vec<i32> {
        let level = current_depth;
        let fractal = &fractal_lattice[parent_fractal_pos];

        let same_level_values: Vec<i32> = vec![
            Fractal::get_right(image_position, fractal.depth - level, global_position_map),
            Fractal::get_down_left(image_position, fractal.depth - level, global_position_map),
            Fractal::get_down_right(image_position, fractal.depth - level, global_position_map),
        ]
        .iter()
        .map(|pos| {
            if let Some(parent_fractal_loc) = global_position_map[level].get(pos) {
                let containing_fractal = &fractal_lattice[parent_fractal_loc];
                let haar_pos = containing_fractal.position_map[level][pos];
                containing_fractal.coefficients[channel][haar_pos].unwrap_or(0)
            } else {
                0
            }
        })
        .collect();

        let below_level_values: Vec<i32> = if multidimensional {
            vec![
                Fractal::get_left(image_position, fractal.depth - level, global_position_map),
                Fractal::get_up_left(image_position, fractal.depth - level, global_position_map),
                Fractal::get_up_right(image_position, fractal.depth - level, global_position_map),
            ]
            .iter()
            .filter_map(|pos| {
                if let Some(parent_fractal_loc) = global_position_map[level].get(pos) {
                    let containing_fractal = &fractal_lattice[parent_fractal_loc];
                    let haar_pos = containing_fractal.position_map[level][pos];
                    let left_neigh_pos = containing_fractal.image_positions[2*haar_pos];
                    let right_neigh_pos = containing_fractal.image_positions[2*haar_pos+1];
                    
                    let haar_current_pos = fractal.position_map[level][&image_position];
                    let left_current_pos = fractal.image_positions[2*haar_current_pos];
                    let right_current_pos = fractal.image_positions[2*haar_current_pos+1];
                    let neighbours = [
                        Fractal::get_right(left_current_pos, fractal.depth - level-1, global_position_map),
                        Fractal::get_down_left(left_current_pos, fractal.depth - level-1, global_position_map),
                        Fractal::get_down_right(left_current_pos, fractal.depth - level-1, global_position_map),
                        Fractal::get_left(left_current_pos, fractal.depth - level-1, global_position_map),
                        Fractal::get_up_left(left_current_pos, fractal.depth - level-1, global_position_map),
                        Fractal::get_up_right(left_current_pos, fractal.depth - level-1, global_position_map),

                        Fractal::get_right(right_current_pos, fractal.depth - level-1, global_position_map),
                        Fractal::get_down_right(right_current_pos, fractal.depth - level-1, global_position_map),
                        Fractal::get_down_left(right_current_pos, fractal.depth - level-1, global_position_map),
                        Fractal::get_left(right_current_pos, fractal.depth - level-1, global_position_map),
                        Fractal::get_up_left(right_current_pos, fractal.depth - level-1, global_position_map),
                        Fractal::get_up_right(right_current_pos, fractal.depth - level-1, global_position_map),
                    ];

                    let first_neigh = if neighbours.contains(&left_neigh_pos) {
                        Some(left_neigh_pos)
                    } else {
                        None
                    };

                    let second_neigh = if neighbours.contains(&right_neigh_pos) {
                        Some(right_neigh_pos)
                    } else {
                        None
                    };

                    Some([
                        first_neigh,
                        second_neigh,
                    ])
                } else {
                    None
                }
            })
            .flatten()
            .filter_map(|x| x)
            .map(|pos| {
                if let Some(parent_fractal_loc) = global_position_map[level+1].get(&pos) {
                    let containing_fractal = &fractal_lattice[parent_fractal_loc];
                    let haar_pos = containing_fractal.position_map[level+1][&pos];
                    containing_fractal.coefficients[channel][haar_pos].unwrap_or(0)
                } else {
                    0
                }
            })
            .collect()
        } else {
            vec![]
        };

       let mut values: Vec<i32> = below_level_values
            .into_iter()
            .chain(same_level_values)
            .collect();

        while values.len() < 7 {
            values.push(0);

        } 
        values
    }

    fn get_image_neighbour_matrices(
        wavelet_image: &WaveletImage,
        global_depth: usize,
        channel: usize,
    ) -> (Vec<DVector<f32>>, Vec<DMatrix<f32>>) {
        let num_parameters = 7;
        let mut matrices: Vec<DMatrix<f32>> = (0..BASE_FRAC_DEPTH)
            .map(|level| {
                DMatrix::<f32>::zeros(
                    wavelet_image.fractal_lattice.len() * (1 << level),
                    num_parameters,
                )
            })
            .collect();

        let mut value_vectors: Vec<DVector<f32>> = (0..BASE_FRAC_DEPTH)
            .map(|level| DVector::<f32>::zeros(wavelet_image.fractal_lattice.len() * (1 << level)))
            .collect();

        let sorted_lattice = wavelet_image.get_sorted_lattice();

        for (i, image_pos) in sorted_lattice[0].iter().enumerate() {
            let fractal = &wavelet_image.fractal_lattice.get(image_pos).unwrap();
            let haar_tree_pos = fractal.position_map[0]
                .get(&image_pos)
                .unwrap();
            if let Some(value) = fractal.coefficients[channel][*haar_tree_pos] {
                let vals = Self::get_neighbour_values(
                    *image_pos,
                    0,
                    image_pos,
                    &wavelet_image.fractal_lattice,
                    &wavelet_image.global_position_map,
                    channel,
                    false,
                );
                value_vectors[0][i] = value as f32;
                for j in 0..vals.len() {
                    matrices[0][(i, j)] = vals[j] as f32;
                }
            }
        }

        for level in (0..global_depth-1) {
            for (i, image_pos) in sorted_lattice[level as usize].iter().enumerate() {
                let parent_pos = &wavelet_image.global_position_map[level as usize][&image_pos];
                let fractal = &wavelet_image.fractal_lattice.get(parent_pos).unwrap();
                let haar_tree_pos = fractal.position_map[level as usize]
                    .get(&image_pos)
                    .unwrap();
                if let Some(value) = fractal.coefficients[channel][*haar_tree_pos] {
                    let vals = Self::get_neighbour_values(
                        *image_pos,
                        level,
                        parent_pos,
                        &wavelet_image.fractal_lattice,
                        &wavelet_image.global_position_map,
                        channel,
                        true,
                    );
                    //dbg!(vals.len());
                    //if vals.len() == 7 {
                    //    dbg!(level, image_pos);
                    //}
                    value_vectors[level as usize + 1][2*i] = fractal.coefficients[channel][2*haar_tree_pos].unwrap_or(0) as f32;
                    value_vectors[level as usize + 1][2*i+1] = fractal.coefficients[channel][2*haar_tree_pos+1].unwrap_or(0) as f32;

                    for j in 0..vals.len() {
                        matrices[level as usize + 1][(2*i, j)] = vals[j] as f32;
                        matrices[level as usize + 1][(2*i +1, j)] = vals[j] as f32;
                    }
                }
            }
        }

        (value_vectors, matrices)
    }

    fn optimize_width_prediction(
        &mut self,
        neighbourhood_matrices: &Vec<DMatrix<f32>>,
        residuals: &Vec<DVector<f32>>,
        global_depth: usize,
        channel: usize,
    ) {
        let mut width_predictors: Vec<[f32; 7]> = vec![];
        for (matrix, residual_vector) in neighbourhood_matrices.iter().zip(residuals.iter()) {
            let mut width_compounds = DMatrix::<f32>::zeros(matrix.nrows(), 7);
            for (i, row) in matrix.row_iter().enumerate() {
                let gradient_horizn = (row[0] - row[3]).abs();
                let gradient_upper_horizn = (row[1] - row[2]).abs();
                let gradient_down_horizn = (row[4] - row[5]).abs();
                let gradient_vert_left = (row[1] - row[5]).abs();
                let gradient_vert_right = (row[2] - row[4]).abs();

                width_compounds[(i, 0)] = 1.0;
                width_compounds[(i, 1)] = gradient_horizn;
                width_compounds[(i, 2)] = gradient_upper_horizn;
                width_compounds[(i, 3)] = gradient_down_horizn;
                width_compounds[(i, 4)] = gradient_vert_left;
                width_compounds[(i, 5)] = gradient_vert_right;
            }

            let least_squares_result = lstsq(&width_compounds, residual_vector, 1e-14).unwrap();
            width_predictors.push(least_squares_result.solution.fixed_rows::<7>(0).into());
        }

        self.width_predictors[channel] = width_predictors;
    }

    fn optimize_value_prediction(
        &mut self,
        neighbourhood_matrices: &Vec<DMatrix<f32>>,
        values: &Vec<DVector<f32>>,
        global_depth: usize,
        channel: usize,
    ) -> Vec<DVector<f32>> {
        let results = neighbourhood_matrices
            .iter()
            .zip(values.iter())
            .map(|(matrix, vector)| lstsq(matrix, vector, 1e-14).unwrap())
            .collect::<Vec<_>>();

        self.value_predictors[channel] = results
            .iter()
            .map(|result| result.solution.fixed_rows::<7>(0).into())
            .collect();

        neighbourhood_matrices
            .iter()
            .zip(values.iter())
            .zip(results.iter())
            .map(|((matrix, value_vector), result)| {
                let prediction = matrix * &result.solution;
                (value_vector - prediction).abs()
            })
            .collect()
    }

    pub fn optimize_parameters(&mut self, wavelet_image: &WaveletImage, channel: usize) {
        let global_depth = wavelet_image.fractal_lattice.values().nth(0).unwrap().depth;

        let (values, matrices) =
            Self::get_image_neighbour_matrices(&wavelet_image, global_depth, channel);

        let residuals = self.optimize_value_prediction(&matrices, &values, global_depth, channel);

        self.optimize_width_prediction(&matrices, &residuals, global_depth, channel);
    }
}

#[cfg(test)]
mod test {
    use crate::images::ImageMetadata;

    use super::*;

    #[test]
    fn unit_test() {
        let wavelet_image = WaveletImage::from_metadata(ImageMetadata::new(10, 10));
        let mut ctx_modeler = ContextModeler::new();
        ctx_modeler.optimize_parameters(&wavelet_image, 0);
    }
}
