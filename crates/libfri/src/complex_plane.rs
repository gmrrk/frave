use std::{arch::asm, backtrace::BacktraceStatus};

use itertools::Itertools;
use num::complex::Complex;
use crate::fractal::{Fractal, BASE_FRAC_DEPTH};

#[derive(Debug)]
pub struct ComplexCell {
    parent_id: usize,
    levels_present: u16,
    haar_tree_position: usize,
}

impl ComplexCell {
    fn new(parent_id: usize, haar_tree_position: usize, level: u32) -> Self {
        ComplexCell {
            parent_id,
            levels_present: 1 << level,
            haar_tree_position,
        }
    }

    fn is_on_level(&self, level: u32) -> bool {
        return (self.levels_present & (1 << level)) == 1 << level;
    }

    fn add_level_presence(&mut self, level: u32) {
        self.levels_present |= 1 << level;
    }
}

pub struct ComplexPlane {
    pub fractals: Vec<Fractal>,
    plane: Vec<Option<ComplexCell>>,
    offset_x: i32,
    offset_y: i32,
    num_rows: usize,
    num_cols: usize,
}

impl ComplexPlane {
    pub fn new(fractals: Vec<Fractal>) -> ComplexPlane {
        let (min_im, max_im) = fractals
            .iter()
            .flat_map(|f| &f.image_positions[..1<<BASE_FRAC_DEPTH])
            .minmax_by(|pos_a, pos_b| pos_a.im.cmp(&pos_b.im))
            .into_option()
            .unwrap();

        let (min_re, max_re) = fractals
            .iter()
            .flat_map(|f| &f.image_positions[..1<<BASE_FRAC_DEPTH])
            .minmax_by(|pos_a, pos_b| pos_a.re.cmp(&pos_b.re))
            .into_option()
            .unwrap();

        let width = (max_re.re - min_re.re + 1) as usize;
        let height = (max_im.im - min_im.im + 1) as usize;

        let mut complex_plane = ComplexPlane {
            plane: std::iter::repeat_with(|| None)
                .take(width * height)
                .collect(),
            offset_x: -0.min(min_re.re),
            offset_y: -0.min(min_im.im),
            num_rows: height,
            num_cols: width,
            fractals: Vec::new(),
        };

        for (id, fractal) in fractals.iter().enumerate() {
            for level in (0..BASE_FRAC_DEPTH).rev() {
                for (haar_pos_offset, pos) in fractal.image_positions[1 << level .. 1 << (level + 1)].iter().enumerate() {
                    let haar_pos = haar_pos_offset + (1 << level);
                    complex_plane.add_cell(*pos, level as u32, id, haar_pos);
                }
            }
        }

        complex_plane.fractals = fractals;

        complex_plane
    }

    pub fn get_minmax_coordinates(&self) -> (i32, i32, i32, i32) {
        let min_re = -self.offset_x;
        let min_im = -self.offset_y;
        let max_re = self.num_cols as i32 + min_re - 1;
        let max_im = self.num_rows as i32 + min_im - 1;
        return (min_re, min_im, max_re, max_im);
    }

    #[inline]
     fn to_index(&self, pos: Complex<i32>) -> Option<usize> {
        let x = self.offset_x + pos.re;
        let y = self.offset_y + pos.im;
        if x < 0 || y < 0 || x >= self.num_cols as i32 || y >= self.num_rows as i32 {
            None
        } else {
            Some(y as usize * self.num_cols + x as usize)
        }
    }

    pub fn add_cell(
        &mut self,
        pos: Complex<i32>,
        level: u32,
        parent_id: usize,
        haar_tree_position: usize,
    ) {
        if let Some(idx) = self.to_index(pos) {
            if let Some(cell) = self.plane[idx].as_mut() {
                cell.add_level_presence(level);
            } else {
                self.plane[idx] = Some(ComplexCell::new(parent_id, haar_tree_position, level));
            }
        }
    }

    pub fn is_cell_on_level(&self, pos: Complex<i32>, level: usize) -> bool {
        self.to_index(pos)
            .and_then(|idx| self.plane[idx].as_ref())
            .map_or(false, |cell| cell.is_on_level(level as u32))
    }

    pub fn get_parent_fractal_at(&self, pos: Complex<i32>, level: usize) -> Option<(&Fractal, usize)> {
        let cell = self.to_index(pos)
            .and_then(|idx| self.plane[idx].as_ref())?;
        let fractal_ref = self.fractals.get(cell.parent_id)?;
        Some((fractal_ref, cell.haar_tree_position >> (BASE_FRAC_DEPTH - level - 1)))
    }

    pub fn get_fractal_mut(&mut self, pos: Complex<i32>) -> Option<&mut Fractal> {
        let parent_id = self.to_index(pos)
            .and_then(|idx| self.plane[idx].as_ref())
            .map(|cell| cell.parent_id)?;
        self.fractals.get_mut(parent_id)
    }
}
