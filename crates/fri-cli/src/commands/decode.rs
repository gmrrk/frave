use image;
use std::fs;
use std::path::PathBuf;
use std::fs::File;
use std::io::BufWriter;

use libfri::decoder::FRIDecoder;


#[derive(clap::Args)]
/// Decodes frave file to bitmap format
pub struct DecodeCommand {
    pub fr_path: PathBuf,

    #[arg(short, default_value_t = String::from("a.bmp"))]
    pub output: String,
}

pub fn decode_image(cmd: DecodeCommand) {
    let data = fs::read(cmd.fr_path).unwrap_or_else(|e| {
        panic!("Failed to open: {e}");
    });

    let decoder = FRIDecoder{ quantization_table: [1;9] };

    match decoder.decode(data) {
        Ok(result) => {
           let img: image::RgbImage = match image::ImageBuffer::from_vec(result.metadata.width as u32, result.metadata.height as u32, result.data) {
                Some(buf) => buf,
                None => {
                    eprintln!("Failed to create image buffer.");
                    return;
                }
            };

            let file = File::create(cmd.output).unwrap();
            let ref mut w = BufWriter::new(file);

            img.write_to(w, image::ImageOutputFormat::Bmp).expect("Failed to write image");
        }
        Err(msg) => println!("Cannot decode, reason: {msg}"),
    }
}
