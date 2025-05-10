use core::fmt;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, BufRead, BufReader, Read};
use std::path::PathBuf;

use libfri::decoder::FRIDecoder;
use libfri::encoder::{EncoderOpts, FRIEncoder};

#[derive(clap::Args)]
pub struct OptimizeCommand {
    pub dataset_path: PathBuf,
}


fn find_arrays(arr: &mut Vec<f32>, n: usize, sum: f32, results: &mut Vec<Vec<f32>>, min_val: f32, max_val: f32) {
    if arr.len() == n {
        if sum == 0. {
            results.push(arr.clone());
        }
        return;
    }

    // Trying each value from min_val to max_val
    for i in min_val as i32..=max_val as i32 {
        if i as f32 <= sum {
            arr.push(i as f32);
            find_arrays(arr, n, sum - i as f32, results, min_val, max_val);
            arr.pop();
        }
    }
}

pub fn optimize(cmd: OptimizeCommand) {
    let paths = fs::read_dir(cmd.dataset_path).expect(&format!("No such directory"));
    fs::create_dir_all("./output").unwrap();
    let mut lowest_mse = f32::MAX;

    let coef_set = [
        [1./6., 1./6., 1./6., 1./6., 1./6., 1./6.],
    ];

    let n = 6;
    let target_sum = 6.;

    let mut results: Vec<Vec<f32>> = Vec::new();
    let mut current_vec: Vec<f32> = Vec::new();
    let min_val: f32 = -3.;
    let max_val: f32 = 3.;

    find_arrays(&mut current_vec, n, target_sum, &mut results, min_val, max_val);

    for arr in results.iter_mut() {
        arr[0] = arr[0] / 6.;
        arr[1] = arr[1] / 6.;
        arr[2] = arr[2] / 6.;
        arr[3] = arr[3] / 6.;
        arr[4] = arr[4] / 6.;
        arr[5] = arr[5] / 6.;
    }
    //dbg!(results);

    let mut best_coef: [f32; 6] = [0.;6];

    for (i, path) in paths.enumerate() {
        if i == 6 {
        let img_path = path.unwrap().path();
        for (j, coefs) in results.iter().enumerate() {
            let img = match image::open(&img_path) {
                Ok(data) => data,
                Err(_) => continue,
            };
            println!("Iter: {}", j);

            let encoder = FRIEncoder::new(EncoderOpts {
                quality: libfri::encoder::EncoderQuality::Lossless,
                emit_coefficients: false,
                verbose: false,
                value_prediction_params: Default::default(), 
                width_prediction_params: Default::default()
            });

            let height = img.height();
            let width = img.width();
            let color = img.color();
            let data = img.into_bytes();

            let frifcolor = match color {
                image::ColorType::L8 => libfri::images::ColorSpace::Luma,
                image::ColorType::Rgb8 => libfri::images::ColorSpace::RGB,
                _ => panic!("Unsupported color scheme for frif image, expected rgb8 or luma8"),
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

            let mut errors: Vec<i32> = Vec::new();

            for i in 0..3 {
                let file = File::open(format!("mse/errors_{}.mse", i)).unwrap();
                let reader = BufReader::new(file);

                for line in reader.lines() {
                    let line = line.unwrap();
                    if let Ok(number) = line.trim().parse::<i32>() {
                        errors.push(number); }
                }

                }
            let mut errors_sorted = errors.clone();
            errors_sorted.sort();
            let median_current = errors_sorted[errors.len()/2];
            let mse_current = errors.iter().sum::<i32>() as f32 /(errors.len() as f32);
            dbg!(mse_current);
            dbg!(median_current);
            dbg!(coefs);
            if mse_current < lowest_mse {
                lowest_mse = mse_current;
                best_coef = coefs[0..6].try_into().unwrap();
            }


            if false {
                fs::write(&img_path, &result)
                    .unwrap_or_else(|e| panic!("Failed to encode frv image: {e}"));
            }
        }
        }
    }
    println!("MSE: {}", lowest_mse);
    println!("Best coefs {:?}", best_coef);
}
