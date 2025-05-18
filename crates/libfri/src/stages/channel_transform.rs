use crate::{encoder::EncoderOpts, images::RasterImage};

// TODO Add YrCrBr
pub fn encode(mut image: RasterImage, encoder_config: &EncoderOpts) -> Result<RasterImage, String> {
    Ok(image)
}

pub fn decode(image: RasterImage) -> Result<RasterImage, String> {
    Ok(image)
}
