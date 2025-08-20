
use crate::images::{CompressedImage, RasterImage};
use crate::stages::wavelet_transform::WaveletImage;
use crate::stages::{channel_transform, entropy_coding, quantization, wavelet_transform, serialize};

enum DecoderStage {
    EncodedImage(Vec<u8>),
    EntropyDecoding(CompressedImage),
    Dequantization(WaveletImage),
    WaveletTransform(WaveletImage),
    ChannelTransform(RasterImage),
    RawImage(RasterImage),
    Failure(String),
}

impl DecoderStage {
    fn forward(self, qnt: &[Option<i32> ;9]) -> DecoderStage {
        match self {
            DecoderStage::EncodedImage(data) => match serialize::decode(data) {
                Ok(result) => DecoderStage::EntropyDecoding(result),
                Err(reason) => DecoderStage::Failure(reason.to_string()),
            }
            DecoderStage::EntropyDecoding(data) => match entropy_coding::decode(data) {
                Ok(result) => DecoderStage::Dequantization(result),
                Err(reason) => DecoderStage::Failure(reason),
            },
            DecoderStage::Dequantization(data) => match quantization::decode(data, qnt) {
                Ok(result) => DecoderStage::WaveletTransform(result),
                Err(reason) => DecoderStage::Failure(reason),
            },
            DecoderStage::WaveletTransform(data) => match wavelet_transform::decode(data) {
                Ok(result) => DecoderStage::ChannelTransform(result),
                Err(reason) => DecoderStage::Failure(reason),
            },
            DecoderStage::ChannelTransform(data) => match channel_transform::decode(data) {
                Ok(result) => DecoderStage::RawImage(result),
                Err(reason) => DecoderStage::Failure(reason),
            },
            other => other,
        }
    }
}

pub struct FRIDecoder {
   pub quantization_table: [Option<i32>; 9],
}

impl FRIDecoder {
    pub fn decode(self, data: Vec<u8>) -> Result<RasterImage, String> {
        let mut stage = DecoderStage::EncodedImage(data);
        while !matches!(stage, DecoderStage::RawImage(_) | DecoderStage::Failure(_)) {
            stage = stage.forward(&self.quantization_table);
        }

        match stage {
            DecoderStage::RawImage(result) => Ok(result),
            DecoderStage::Failure(msg) => Err(String::from("Failed to decode: ".to_owned() + &msg)),
            _ => unreachable!(),
        }
    }
}
