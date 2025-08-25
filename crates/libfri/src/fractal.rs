use std::collections::HashMap;

use num::complex::Complex;


//fn get_literals<const N: usize>(d: f32) -> [Complex<f32>; N] {
//    let base = Complex::new(d / 2., (2. - (d / 2.).powf(2.)).sqrt());
//    let mut powers = [base; N];
//    let mut i = 1;
//
//    let mut pow = Complex::<f32> { re: 1., im: 0. };
//    while i < N {
//        powers[i] = Complex::<f32> {
//            re: (-pow.re / base.re).round(),
//            im: (pow.im / base.im).round(),
//        };
//        pow *= base;
//        i += 1;
//    }
//
//    powers[0] = Complex::<f32> { re: 0., im: 1. };
//
//    powers.swap(1, 2);
//
//    powers
//}

pub const BASE_FRAC_DEPTH: usize = 9;

pub static LITERALS: [Complex<i32>; 30] = 
             [
               Complex { re: 0, im: 1 },
               Complex { re: -1, im: 1 },
               Complex { re: 2, im: 0 },
               Complex { re: -3, im: -1 },
               Complex { re: 5, im: -1 },
               Complex { re: 1, im: 3 },
               Complex { re: -11, im: -1 },
               Complex { re: 9, im: -5 },
               Complex { re: 13, im: 7 },
               Complex { re: -31, im: 3 },
               Complex { re: 5, im: -17 },
               Complex { re: 57, im: 11 },
               Complex { re: -67, im: 23 },
               Complex { re: -47, im: -45 },
               Complex { re: 181, im: -1 },
               Complex { re: -87, im: 91 },
               Complex { re: -275, im: -89 },
               Complex { re: 449, im: -93 },
               Complex { re: 101, im: 271 },
               Complex { re: -999, im: -85 },
               Complex { re: 797, im: -457 },
               Complex { re: 1201, im: 627 },
               Complex { re: -2795, im: 287 },
               Complex { re: 393, im: -1541 },
               Complex { re: 5197, im: 967 },
               Complex { re: -5983, im: 2115 },
               Complex { re: -4411, im: -4049 },
               Complex { re: 16377, im: -181 },
               Complex { re: -7555, im: 8279 },
               Complex {
                    re: -25199,
                    im: -7917,
                },
            ];

#[derive(Debug)]
pub struct Fractal {
    pub depth: usize,
    pub center: Complex<i32>,
    pub coefficients: [Vec<Option<i32>>; 3],
    pub parameter_predictors: [Vec<(usize, i32)>; 3],
    pub values: [Vec<Option<i32>>; 3],
    pub position_map: Vec<HashMap<Complex<i32>, usize>>,
    pub image_positions: Vec<Complex<i32>>,
}

impl Fractal {
    pub fn new(depth: usize, center: Complex<i32>) -> Self {
        let mut position_map = vec![HashMap::new(); depth];
        let mut image_positions = vec![Complex::<i32>::new(0, 0); 1 << (depth + 1)];
        image_positions[0] = center;
        image_positions[1] = center;
        for level in 0..depth {
            for pos in 1 << level..1 << (level + 1) {
                position_map[level].insert(image_positions[pos], pos);
                image_positions[2 * pos] = image_positions[pos];
                image_positions[2 * pos + 1] =
                    image_positions[pos] + LITERALS[depth - level - 1];
            }
        }

        Fractal {
            depth,
            center,
            coefficients: [vec![], vec![], vec![]],
            parameter_predictors: [
                vec![(0, 0); 1 << depth],
                vec![(0, 0); 1 << depth],
                vec![(0, 0); 1 << depth],
            ],
            position_map,
            image_positions,
            values: [vec![], vec![], vec![]],
        }
    }

    pub fn get_nearby_vectors(depth: usize) -> [Complex<i32>; 6] {
        if depth == 1 {
            let zl = Complex::new(-1, 1);
            let zmd = Complex::new(0, 2);
            [zl, zl - zmd, -zmd, -zl, zmd - zl, zmd]
        } else if depth == 2 {
            let zl = Complex::new(-2, 0);
            let zmd = Complex::new(-0, -2);
            return [zl, zl - zmd, -zmd, -zl, zmd - zl, zmd];
        } else if depth == 3 {
            let zl = Complex::new(-3, -1);
            let zmd = Complex::new(-1, -3);
            return [zl, zl - zmd, -zmd, -zl, zmd - zl, zmd];
        } else {
            let zl = LITERALS[depth];
            let zmd = LITERALS[depth + 1] + zl;

            return [zl, zl - zmd, -zmd, -zl, zmd - zl, zmd];
        }
    }

    pub fn get_neighbour_locations(&self) -> [Complex<i32>; 6] {
        let vectors = Self::get_nearby_vectors(self.depth);
        vectors.map(|x| self.center + x).try_into().unwrap()
    }

    pub fn get_left(
        center: Complex<i32>,
        depth: usize,
        _global_position_map: &[HashMap<Complex<i32>, Complex<i32>>; BASE_FRAC_DEPTH],
    ) -> Complex<i32> {
        let vectors = Self::get_nearby_vectors(depth);
        center + vectors[4]
    }

    pub fn get_right(
        center: Complex<i32>,
        depth: usize,
        _global_position_map: &[HashMap<Complex<i32>, Complex<i32>>; BASE_FRAC_DEPTH],
    ) -> Complex<i32> {
        let vectors = Self::get_nearby_vectors(depth);
        center + vectors[1]
    }

    pub fn get_down_left(
        center: Complex<i32>,
        depth: usize,
        global_position_map: &[HashMap<Complex<i32>, Complex<i32>>; BASE_FRAC_DEPTH],
    ) -> Complex<i32> {
        let vectors = Self::get_nearby_vectors(depth);
        if depth == 2
            && !global_position_map[BASE_FRAC_DEPTH - depth].contains_key(&(center + vectors[3]))
            && global_position_map[BASE_FRAC_DEPTH - depth].contains_key(&(center + Complex::new(1, 1)))
        {
            center + Complex::new(1, 1)
        } else {
            center + vectors[3]
        }
    }

    pub fn get_down_right(
        center: Complex<i32>,
        depth: usize,
        global_position_map: &[HashMap<Complex<i32>, Complex<i32>>; BASE_FRAC_DEPTH],
    ) -> Complex<i32> {
        let vectors = Self::get_nearby_vectors(depth);
        if depth == 2
            && !global_position_map[BASE_FRAC_DEPTH - depth].contains_key(&(center + vectors[3]))
            && global_position_map[BASE_FRAC_DEPTH - depth].contains_key(&(center + Complex::new(1, 1)))
        {
            center + Complex::new(1, 1) + vectors[1]
        } else {
            center + vectors[2]
        }
    }

    pub fn get_up_right(
        center: Complex<i32>,
        depth: usize,
        global_position_map: &[HashMap<Complex<i32>, Complex<i32>>; BASE_FRAC_DEPTH],
    ) -> Complex<i32> {
        let vectors = Self::get_nearby_vectors(depth);
        if depth == 2
            && !global_position_map[BASE_FRAC_DEPTH - depth].contains_key(&(center + vectors[0]))
            && global_position_map[BASE_FRAC_DEPTH - depth].contains_key(&(center + Complex::new(-1, -1)))
        {
            center + Complex::new(-1, -1)
        } else {
            center + vectors[0]
        }
    }

    pub fn get_up_left(
        center: Complex<i32>,
        depth: usize,
        global_position_map: &[HashMap<Complex<i32>, Complex<i32>>; BASE_FRAC_DEPTH],
    ) -> Complex<i32> {
        let vectors = Self::get_nearby_vectors(depth);
        if depth == 2
            && !global_position_map[BASE_FRAC_DEPTH - depth].contains_key(&(center + vectors[0]))
            && global_position_map[BASE_FRAC_DEPTH - depth].contains_key(&(center + Complex::new(-1, -1)))
        {
            center + Complex::new(-1, -1) + vectors[4]
        } else {
            center + vectors[5]
        }
    }

}

