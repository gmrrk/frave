use crate::images::{FractalVariant, ColorSpace, CompressedImage, RasterImage, ImageMetadata};
use crate::stages::entropy_coding::AnsContext;
use crate::stages::wavelet_transform::WaveletImage;
use crate::stages::{channel_transform, entropy_coding, prediction, quantization, serialize, wavelet_transform};

enum EncoderStage {
    RawImage(RasterImage),
    ChannelTransform(RasterImage),
    WaveletTransform(RasterImage),
    Quantization(WaveletImage),
    Prediction(WaveletImage),
    EntropyEncoding(WaveletImage, [Vec<AnsContext>; 3]),
    EncodedImage(CompressedImage),
    SerializedImage(Vec<u8>),
    Failure(String),
}

impl EncoderStage {
    fn forward(self, encoder_options: &mut EncoderOpts) -> EncoderStage {
        match self {
            EncoderStage::RawImage(data) => EncoderStage::ChannelTransform(data),
            EncoderStage::ChannelTransform(data) => match channel_transform::encode(data) {
                Ok(result) => EncoderStage::WaveletTransform(result),
                Err(reason) => EncoderStage::Failure(reason),
            },
            EncoderStage::WaveletTransform(data) => match wavelet_transform::encode(data, encoder_options) {
                Ok(result) => EncoderStage::Quantization(result),
                Err(reason) => EncoderStage::Failure(reason),
            },
            EncoderStage::Quantization(data) => match quantization::encode(data) {
                Ok(result) => EncoderStage::Prediction(result),
                Err(reason) => EncoderStage::Failure(reason),
            },
            EncoderStage::Prediction(mut data) => match prediction::encode(&mut data, encoder_options) {
                Ok(result) => EncoderStage::EntropyEncoding(data, result),
                Err(reason) => EncoderStage::Failure(reason),
            },
            EncoderStage::EntropyEncoding(data, contexts) => match entropy_coding::encode(data, contexts, encoder_options) {
                Ok(result) => EncoderStage::EncodedImage(result),
                Err(reason) => EncoderStage::Failure(reason),
            },
            EncoderStage::EncodedImage(data) => match serialize::encode(data) {
                Ok(result) => EncoderStage::SerializedImage(result),
                Err(reason) => EncoderStage::Failure(reason.to_string())
            }
            other => other,
        }
    }
}

pub enum EncoderQuality {
    Low,
    Medium,
    High,
    Lossless,
}

pub struct EncoderOpts {
   pub quality: EncoderQuality,
   pub emit_coefficients: bool,
   pub value_prediction_params: [Vec<[f32; 6]>; 4],
   pub width_prediction_params: [Vec<[f32; 6]>; 4],
   pub verbose: bool,
}

pub struct FRIEncoder {
    opts: EncoderOpts,
}

impl Default for EncoderOpts {
    fn default() -> Self {
        Self {
            emit_coefficients: false,
            quality: EncoderQuality::Lossless,
            value_prediction_params: Default::default(),
            width_prediction_params: Default::default(),
            verbose: false,
        }
    }
}

impl FRIEncoder {
    pub fn new(opts: EncoderOpts) -> FRIEncoder {
        FRIEncoder{ opts }
    }

    pub fn encode(
        mut self,
        data: Vec<u8>,
        height: u32,
        width: u32,
        colorspace: ColorSpace,
    ) -> Result<Vec<u8>, String> {
        let image = RasterImage {
            data,
            metadata: ImageMetadata{height, width, colorspace, variant: FractalVariant::TameTwindragon}
        };

        let mut stage = EncoderStage::RawImage(image);
        while !matches!(stage, EncoderStage::SerializedImage(_) | EncoderStage::Failure(_)) {
            stage = stage.forward(&mut self.opts);
        }

        match stage {
            EncoderStage::SerializedImage(image) => Ok(image),
            EncoderStage::Failure(msg) => Err(String::from("Failed to decode: ".to_owned() + &msg)),
            _ => unreachable!(),
        }
    }
}
