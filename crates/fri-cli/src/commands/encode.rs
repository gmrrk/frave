use std::fs;
use std::path::PathBuf;

use libfri::encoder::{EncoderOpts, FRIEncoder};

#[derive(clap::Args)]
/// Encodes bitmap file to frave format
pub struct EncodeCommand {
    pub bmp_path: PathBuf,

    #[arg(short, default_value_t = String::from("a.frv"))]
    pub output: String,

    #[arg(long, default_value_t = 100)]
    pub quality: u8,

    #[arg(long, default_value_t = false)]
    pub emit_coefficients: bool,
}

pub fn encode_image(cmd: EncodeCommand, verbose: bool) {
    let img = image::open(cmd.bmp_path).unwrap_or_else(|e| {
        panic!("Failed to open: {e}");
    });

    let luma_img = img;
    let encoder = FRIEncoder::new(EncoderOpts {
        emit_coefficients: cmd.emit_coefficients,
        ..Default::default() 
    });

    let height = luma_img.height();
    let width = luma_img.width();
    let color = luma_img.color();
    let data = luma_img.into_bytes();

    let frifcolor = match color {
        image::ColorType::L8 => libfri::images::ColorSpace::Luma,
        image::ColorType::Rgb8 => libfri::images::ColorSpace::RGB,
        _ => panic!("Unsupported color scheme for frif image, expected rgb8 or luma8"),
    };
    let uncompressed_size = data.len();

    match encoder.encode(data, height, width, frifcolor, cmd.quality) {
        Ok(result) => {
            if verbose {
                println!("Before compression size: {}", uncompressed_size);
                println!("After compression size: {}", result.len());
                println!("Compression rate: {}%", (uncompressed_size as f32 - result.len() as f32)/uncompressed_size as f32 * 100.);
            }

            fs::write(cmd.output, result).unwrap_or_else(|e| panic!("Failed to encode frv image: {e}"))
        }
        Err(msg) => eprintln!("Cannot encode, reason: {msg}"),
    }
}
