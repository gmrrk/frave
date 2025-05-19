use crate::fractal::LITERALS;
use crate::stages::entropy_coding::AnsContext;
use crate::stages::serialize::SerializeError;
use crate::stages::wavelet_transform::WaveletImage;
use num::complex::Complex;

#[derive(Debug, Clone)]
pub enum ColorSpace {
    Luma,
    YCbCr,
    RGB,
}

impl ColorSpace {
    pub fn num_channels(&self) -> usize {
        match self {
            ColorSpace::Luma => 1,
            ColorSpace::YCbCr => 3,
            ColorSpace::RGB => 3,
        }
    }

    pub fn get_encoding(&self) -> u32 {
        match self {
            ColorSpace::Luma => 0b01,
            ColorSpace::RGB => 0b10,
            ColorSpace::YCbCr => 0b11,
        }
    }

    pub fn from_encoding(val: u8) -> Result<ColorSpace, SerializeError> {
        match val {
            0b01 => Ok(ColorSpace::Luma),
            0b10 => Ok(ColorSpace::RGB),
            0b11 => Ok(ColorSpace::YCbCr),
            _ => Err(SerializeError::InvalidMetadata),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FractalVariant {
    TameTwindragon,
    Twindragon,
    Boxes,
}

impl FractalVariant {
    pub fn get_encoding(&self) -> u32 {
        match self {
            FractalVariant::TameTwindragon => 0b01,
            FractalVariant::Twindragon => 0b10,
            FractalVariant::Boxes => 0b11,
        }
    }

    pub fn from_encoding(val: u8) -> Result<FractalVariant, SerializeError> {
        match val {
            0b01 => Ok(FractalVariant::TameTwindragon),
            0b10 => Ok(FractalVariant::Twindragon),
            0b11 => Ok(FractalVariant::Boxes),
            _ => Err(SerializeError::InvalidMetadata),
        }
    }
}

#[derive(Clone)]
pub struct ImageMetadata {
    pub height: u32,
    pub width: u32,
    pub colorspace: ColorSpace,
    pub variant: FractalVariant,
}

impl ImageMetadata {
    pub fn new(height: u32, width: u32) -> Self {
        ImageMetadata { height, width, colorspace: ColorSpace::RGB, variant: FractalVariant::TameTwindragon }
    }
}

#[derive(Clone)]
pub struct RasterImage {
    pub metadata: ImageMetadata,
    pub data: Vec<u8>,
}

impl RasterImage {
    #[inline]
    pub fn get_pixel(&self, x: i32, y: i32, channel: usize) -> Option<i32> {
        if x >= 0 && y >= 0 && x < self.metadata.width as i32 && y < self.metadata.height as i32 {
            let (x, y): (u32, u32) = (x.try_into().unwrap(), y.try_into().unwrap());
            let num_channels = self.metadata.colorspace.num_channels() as u32;
            // TODO: Add different subsamplings, currently 4:4:4 is supported
            let position = ((y * self.metadata.width + x) * num_channels + channel as u32) as usize;
            let value = self.data[position];
            Some(i32::from(value))
        } else {
            None
        }
    }

    #[inline]
    pub fn set_pixel(&mut self, x: i32, y: i32, value: i32, channel: usize) {
        if x >= 0 && y >= 0 && x < self.metadata.width as i32 && y < self.metadata.height as i32 {
            let (x, y): (u32, u32) = (x.try_into().unwrap(), y.try_into().unwrap());
            let num_channels = self.metadata.colorspace.num_channels() as u32;
            // TODO: Add different subsamplings, currently 4:4:4 is supported
            let position = ((y * self.metadata.width + x) * num_channels + channel as u32) as usize;
            self.data[position] = value.clamp(0, 255) as u8;
        }
    }
}

pub struct ChannelData {
   pub ans_contexts: Vec<AnsContext>,
   pub data: Vec<u8>,
   pub value_prediction_parameters: Vec<[f32;7]>,
   pub width_prediction_parameters: Vec<[f32;7]>,
}

pub struct CompressedImage {
    pub metadata: ImageMetadata,
    pub channel_data: [Option<ChannelData>; 3],
}
