use core::{f32, fmt};
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read};
use std::path::PathBuf;

use itertools::Itertools;
use ssimulacra2::{
    compute_frame_ssimulacra2, Blur, ColorPrimaries, Frame, MatrixCoefficients, Plane,
    TransferCharacteristic, Yuv, YuvConfig,
};

use libfri::decoder::FRIDecoder;
use libfri::encoder::{EncoderOpts, FRIEncoder};
use yuvxyb::LinearRgb;

#[derive(clap::Args)]
pub struct OptimizeCommand {
    pub dataset_path: PathBuf,
}

fn loss_with_quant(comp: f64, ssim2: f64) -> f64 {
    comp*comp*comp * ssim2
}

pub fn optimize(cmd: OptimizeCommand) {
    let paths = fs::read_dir(cmd.dataset_path).expect(&format!("No such directory"));
    fs::create_dir_all("./output").unwrap();
    let mut quant_table = [None; 9];

    for (i, path) in paths.enumerate() {
        if i != 0 {
            let img_path = path.unwrap().path();
            dbg!(&img_path);
            let img = match image::open(&img_path) {
                Ok(data) => data,
                Err(_) => continue,
            };
            for j in (0..9).rev().chain((0..9).rev()) {
                let mut max_gain: f64 = f64::MIN;

                println!("Optimizing {} position", j);
                for k in [None, Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7), Some(8), Some(9)] {
                    let mut enc_qnt_table = quant_table.clone();
                    enc_qnt_table[j] = k;

                    let encoder = FRIEncoder::new(EncoderOpts {
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
                    let data = c_img.clone().into_bytes();
                    let lin_grad_data: Vec<[u8; 3]> = c_img
                        .into_bytes()
                        .chunks_exact(3)
                        .map(|x| x.try_into().unwrap())
                        .collect();
                    let lin_rgb_input = LinearRgb::new(
                        lin_grad_data
                            .into_iter()
                            .map(|e| [e[0] as f32, e[1] as f32, e[2] as f32])
                            .collect(),
                        width as usize,
                        height as usize,
                    ).unwrap();

                    let frifcolor = match color {
                        image::ColorType::L8 => libfri::images::ColorSpace::Luma,
                        image::ColorType::Rgb8 => libfri::images::ColorSpace::RGB,
                        _ => panic!(
                            "Unsupported color scheme for frif image, expected rgb8 or luma8"
                        ),
                    };
                    let result = encoder
                        .encode(data, height, width, frifcolor, 50)
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
                                decoded.data.into_iter().map(|x| x as u8).collect(),
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
                            let decoded_img2 = img.bytes();
                            let len = original_img.len() as f32;

                            let lin_rgb_output = LinearRgb::new(
                                decoded_img2
                                    .into_iter()
                                    .chunks(3)
                                    .into_iter()
                                    .map(|mut e| {
                                        [
                                            e.next().unwrap().unwrap() as f32,
                                            e.next().unwrap().unwrap() as f32,
                                            e.next().unwrap().unwrap() as f32,
                                        ]
                                    })
                                    .collect(),
                                width as usize,
                                height as usize,
                            ).unwrap();

                            let mut mse: f32 = 0.;
                            for (x, y) in decoded_img.into_iter().zip(original_img.into_iter()) {
                                mse += (x.unwrap() - y).pow(2) as f32;
                            }
                            let ssim2 = compute_frame_ssimulacra2(lin_rgb_input, lin_rgb_output).unwrap();
                            mse = mse / len;
                            let comp = (1. - compressed_lenght as f64 / len as f64) * 100.;
                            println!(
                                "SSIM2: {}, mse: {}, COMP_LENGTH: {}, LOSS: {}",
                                ssim2,
                                mse as f32,
                                compressed_lenght,
                                loss_with_quant(comp, ssim2)
                            );

                            if loss_with_quant(comp, ssim2) >= max_gain {
                                quant_table[j] = k;
                                max_gain = loss_with_quant(comp, ssim2);
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
