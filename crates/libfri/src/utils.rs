use std::cmp::Ordering;

use num::Complex;

pub const fn get_prev_power_two(x: usize) -> usize {
    let mut num = x;
    num |= num >> 1;
    num |= num >> 2;
    num |= num >> 4;
    num |= num >> 8;
    num |= num >> 16;

    num ^ (num >> 1)
}


pub fn order_complex<T: std::cmp::PartialEq + std::cmp::PartialOrd>(
    a: &Complex<T>,
    b: &Complex<T>,
) -> Ordering {
    if a.im > b.im {
        Ordering::Greater
    } else if a.im < b.im {
        Ordering::Less
    } else if a.im == b.im && a.re > b.re {
        Ordering::Greater
    } else if a.im == b.im && a.re < b.re {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

pub fn pack_signed(k: i32) -> u32 {
    if k >= 0 {
        2 * k as u32
    } else {
        (-2 * k - 1) as u32
    }
}

pub fn unpack_signed(k: u32) -> i32 {
    if k % 2 == 0 {
        (k / 2) as i32
    } else {
        (k + 1) as i32 / -2
    }
}
