use num::Float;

use crate::images::{ColorSpace, RasterImage};


fn y_cb_cr_lossy_transform(mut image: RasterImage) -> Result<RasterImage, String> {
    for i in 0..image.metadata.width {
        for j in 0..image.metadata.height {
            let r = image.get_pixel(i as i32, j as i32, 0).unwrap_or(0);
            let g = image.get_pixel(i as i32, j as i32, 1).unwrap_or(0);
            let b = image.get_pixel(i as i32, j as i32, 2).unwrap_or(0);

            let y = 0. + 0.299 * r as f32 + 0.587 * g as f32+ 0.114 * b as f32;
            let cb = -0.168736 * r as f32 - 0.331264*g as f32 + 0.5*b as f32;
            let cr =  0.5 * r as f32 - 0.418688 * g as f32 - 0.081312 * b as f32;
            image.set_pixel(i as i32, j as i32, y.round() as i32, 0);
            image.set_pixel(i as i32, j as i32, cb.round() as i32, 1);
            image.set_pixel(i as i32, j as i32, cr.round() as i32, 2);
        }
    }

    Ok(image)
}

fn y_cb_cr_lossy_transform_reverse(mut image: RasterImage) -> Result<RasterImage, String> {
    for i in 0..image.metadata.width {
        for j in 0..image.metadata.height {
            let y = image.get_pixel(i as i32, j as i32, 0).unwrap_or(0);
            let cb = image.get_pixel(i as i32, j as i32, 1).unwrap_or(0);
            let cr = image.get_pixel(i as i32, j as i32, 2).unwrap_or(0);

            let r = y as f32 + 1.402*(cr as f32);
            let g = y as f32 - 0.34413 * (cb as f32) - 0.71414*(cr as f32);
            let b = y as f32 + 1.772*(cb as f32);

            image.set_pixel(i as i32, j as i32, r.round().clamp(0., 255.) as i32, 0);
            image.set_pixel(i as i32, j as i32, g.round().clamp(0., 255.) as i32, 1);
            image.set_pixel(i as i32, j as i32, b.round().clamp(0., 255.) as i32, 2);
        }
    }
    Ok(image)
}

fn y_cb_cr_lossless_transform(mut image: RasterImage) -> Result<RasterImage, String> {
    for i in 0..image.metadata.width {
        for j in 0..image.metadata.height {
            let r = image.get_pixel(i as i32, j as i32, 0).unwrap_or(0);
            let g = image.get_pixel(i as i32, j as i32, 1).unwrap_or(0);
            let b = image.get_pixel(i as i32, j as i32, 2).unwrap_or(0);

            let y = ((r + 2*g + b) as f32 / 4.).floor();
            let cb = b - g;
            let cr = r - g; 

            image.set_pixel(i as i32, j as i32, y as i32, 0);
            image.set_pixel(i as i32, j as i32, cb as i32, 1);
            image.set_pixel(i as i32, j as i32, cr as i32, 2);
        }
    }

    Ok(image)

}

fn y_cb_cr_lossless_transform_reverse(mut image: RasterImage) -> Result<RasterImage, String> {
    for i in 0..image.metadata.width {
        for j in 0..image.metadata.height {
            let y = image.get_pixel(i as i32, j as i32, 0).unwrap_or(0);
            let cb = image.get_pixel(i as i32, j as i32, 1).unwrap_or(0);
            let cr = image.get_pixel(i as i32, j as i32, 2).unwrap_or(0);

            let g = (y as f32 - ((cb + cr) as f32/4.).floor()) as i32;
            let r = cr + g as i32;
            let b = cb + g as i32;

            image.set_pixel(i as i32, j as i32, r, 0);
            image.set_pixel(i as i32, j as i32, g, 1);
            image.set_pixel(i as i32, j as i32, b, 2);
        }
    }

    Ok(image)
}

fn center_values(mut image: RasterImage) -> RasterImage {
    for i in 0..image.metadata.width {
        for j in 0..image.metadata.height {
            for c in 0..image.metadata.colorspace.num_channels() {
                let val = image.get_pixel(i as i32, j as i32, c).unwrap_or(0);
                image.set_pixel(i as i32, j as i32, val - 127, c);
            }
        }
    }
    image
}

fn reverse_center_values(mut image: RasterImage) -> RasterImage {
    for i in 0..image.metadata.width {
        for j in 0..image.metadata.height {
            for c in 0..image.metadata.colorspace.num_channels() {
                let val = image.get_pixel(i as i32, j as i32, c).unwrap_or(0);
                image.set_pixel(i as i32, j as i32, val + 127, c);
            }
        }
    }
    image
}

pub fn encode(mut image: RasterImage) -> Result<RasterImage, String> {
    if image.metadata.colorspace == ColorSpace::Luma {
        return Ok(image)
    }

    match &image.metadata.quality{
        100 => y_cb_cr_lossless_transform(image),
        _other => y_cb_cr_lossy_transform(image),

    }
}

pub fn decode(mut image: RasterImage) -> Result<RasterImage, String> {
    if image.metadata.colorspace == ColorSpace::Luma {
        return Ok(image)
    }

    match image.metadata.quality {
        100 => y_cb_cr_lossless_transform_reverse(image),
        _other => y_cb_cr_lossy_transform_reverse(image),

    }
}
