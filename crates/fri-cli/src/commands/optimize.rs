use core::{f32, fmt};
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read};
use std::path::PathBuf;

use libfri::decoder::FRIDecoder;
use libfri::encoder::{EncoderOpts, FRIEncoder};

#[derive(clap::Args)]
pub struct OptimizeCommand {
    pub dataset_path: PathBuf,
}

fn loss_with_quant(length: usize, mse: f32) -> f32 {
    let x = length as f32 / 1000.;
    (x*x) * (mse +1.).sqrt().sqrt()
}

pub fn optimize(cmd: OptimizeCommand) {
    let paths = fs::read_dir(cmd.dataset_path).expect(&format!("No such directory"));
    fs::create_dir_all("./output").unwrap();
    let mut quant_table = [1; 9];

    for (i, path) in paths.enumerate() {
        if i != 0 {
            let img_path = path.unwrap().path();
            dbg!(&img_path);
            let img = match image::open(&img_path) {
                Ok(data) => data,
                Err(_) => continue,
            };
            for j in (0..9).rev().chain(1..9) {
                let mut min_loss: f32 = f32::MAX;
                println!("Optimizing {} position", j);
                for k in 1..32 {
                    let mut enc_qnt_table = quant_table.clone();
                    enc_qnt_table[j] = k;

                    let encoder = FRIEncoder::new(EncoderOpts {
                        quality: libfri::encoder::EncoderQuality::Lossless,
                        emit_coefficients: false,
                        verbose: false,
                        value_prediction_params: Default::default(),
                        width_prediction_params: Default::default(),
                        quantization_table: enc_qnt_table,
                    });

                    let c_img = img.clone();

                    let height = c_img.height();
                    let width = c_img.width();
                    let color = c_img.color();
                    let data = c_img.into_bytes();

                    let frifcolor = match color {
                        image::ColorType::L8 => libfri::images::ColorSpace::Luma,
                        image::ColorType::Rgb8 => libfri::images::ColorSpace::RGB,
                        _ => panic!(
                            "Unsupported color scheme for frif image, expected rgb8 or luma8"
                        ),
                    };
                    let result = encoder
                        .encode(data, height, width, frifcolor)
                        .unwrap_or_else(|e| {
                            panic!(
                                "Cannot encode {}, reason: {}",
                                img_path.file_name().unwrap().to_str().unwrap(),
                                e
                            )
                        });

                    let compressed_lenght = result.len();
                    let decoder = FRIDecoder {
                        quantization_table: enc_qnt_table,
                    };
                    match decoder.decode(result) {
                        Ok(decoded) => {
                            let img: image::RgbImage = match image::ImageBuffer::from_vec(
                                decoded.metadata.width as u32,
                                decoded.metadata.height as u32,
                                decoded.data,
                            ) {
                                Some(buf) => buf,
                                None => {
                                    eprintln!("Failed to create image buffer.");
                                    return;
                                }
                            };

                            let mut output_path: PathBuf = PathBuf::from(r"./output/");
                            output_path.push(img_path.file_name().unwrap());
                            output_path.set_extension("bmp");
                            let file = File::create(&output_path).unwrap();
                            let ref mut w = BufWriter::new(file);

                            img.write_to(w, image::ImageOutputFormat::Bmp)
                                .expect("Failed to write image");

                            let original_img = match image::open(&img_path) {
                                Ok(data) => data.into_bytes(),
                                Err(_) => continue,
                            };
                            let decoded_img = img.bytes();
                            let len = original_img.len() as f32;

                            let mut mse: f32 = 0.;
                            for (x, y) in decoded_img.into_iter().zip(original_img.into_iter()) {
                                mse += (x.unwrap() - y).pow(2) as f32;
                            }
                            mse = mse / len;
                            println!("MSE: {}, COMP_LENGTH: {}, LOSS: {}", mse as f32, compressed_lenght, loss_with_quant(compressed_lenght, mse));
                            if loss_with_quant(compressed_lenght, mse) < min_loss {
                                quant_table[j] = k;
                                min_loss = loss_with_quant(compressed_lenght, mse);
                            }
                        }
                        Err(msg) => println!("Cannot decode, reason: {msg}"),
                    }
                }
                println!(" quant {:?}", quant_table);
            }
          println!("Best quant {:?}", quant_table);
          println!("")
        }
    }
}
