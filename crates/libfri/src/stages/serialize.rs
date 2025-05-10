use itertools::Itertools;
use num::traits::ToBytes;
use std::array::TryFromSliceError;
use std::error::Error;
use std::fmt::Display;
use std::mem;

use crate::images::{ChannelData, ColorSpace, CompressedImage, FractalVariant, ImageMetadata};
use crate::stages::entropy_coding::{AnsContext, ALPHABET_SIZE};

#[derive(Debug)]
pub enum SerializeError {
    InvalidSignature,
    InvalidMetadata,
    MalformedImageBytes,
    SliceConversion(TryFromSliceError),
}

impl Display for SerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use SerializeError::*;
        let var_name = match self {
            InvalidSignature => write!(f, "Invalid signature for FRIF image."),
            SliceConversion(e) => write!(f, "Slice from input bytes is malformed: {}", e),
            InvalidMetadata => write!(f, "Invalid metadata"),
            MalformedImageBytes => write!(f, "Malformed image bytes"),
        };
        var_name
    }
}

impl From<TryFromSliceError> for SerializeError {
    fn from(err: TryFromSliceError) -> Self {
        SerializeError::SliceConversion(err)
    }
}
impl Error for SerializeError {}

#[allow(non_snake_case, non_upper_case_globals)]
mod Segments {
    pub const EHD: &[u8] = &[0xFF, 0xB2]; // Entropy Header Data
    pub const DAT: &[u8] = &[0xFF, 0xB4]; // Data
    pub const EOC: &[u8] = &[0xFF, 0xB8]; // End Of Channel
    pub const PRD: &[u8] = &[0xFF, 0xBB]; // Prediction params
    pub const EOI: &[u8] = &[0xFF, 0xDF]; // End Of Image
}

pub fn encode(mut image: CompressedImage) -> Result<Vec<u8>, SerializeError> {
    let mut serial = Vec::new();
    serial.reserve(mem::size_of_val(&image));

    serial.extend_from_slice(b"frif");
    serial.extend_from_slice(&image.metadata.height.to_le_bytes());
    serial.extend_from_slice(&image.metadata.width.to_le_bytes());

    let mut mdat: u32 = 0x0;

    // colorspace
    let colorspace = &image.metadata.colorspace.get_encoding();
    mdat |= colorspace << 30;

    // variant
    let variant = &image.metadata.variant.get_encoding();
    mdat |= variant << 28;

    serial.extend_from_slice(&mdat.to_le_bytes());

    let mut i = 0;
    while let Some(ChannelData {
        ans_contexts,
        data,
        value_prediction_parameters,
        width_prediction_parameters,
    }) = &image.channel_data[i].take()
    {
        i += 1;

        serial.extend_from_slice(Segments::PRD);
        serial.extend_from_slice(
            &value_prediction_parameters
                .iter()
                .flat_map(|s| s.iter().flat_map(|x| x.to_le_bytes()))
                .collect::<Vec<u8>>(),
        );

        serial.extend_from_slice(
            &width_prediction_parameters
                .iter()
                .flat_map(|s| s.iter().flat_map(|x| x.to_le_bytes()))
                .collect::<Vec<u8>>(),
        );

        for ctx in ans_contexts {
            serial.extend_from_slice(Segments::EHD);
            serial.extend_from_slice(
                &(ctx.max_freq_bits).to_le_bytes(),
            );
            serial.extend_from_slice(&(ctx.off_distribution_values.len().to_le_bytes()));
            serial.extend_from_slice(
                &ctx.off_distribution_values
                    .iter()
                    .flat_map(|s| s.to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
        }
        serial.extend_from_slice(Segments::DAT);
        serial.extend_from_slice(&data.len().to_le_bytes());
        serial.extend(data);
        serial.extend_from_slice(Segments::EOC);
        if i >= image.metadata.colorspace.num_channels() {
            break;
        }
    }

    serial.extend_from_slice(Segments::EOI);
    return Ok(serial);
}

pub fn decode(bytes: Vec<u8>) -> Result<CompressedImage, SerializeError> {
    let mut offset = 0;
    if &bytes[offset..offset + 4] != b"frif" {
        return Err(SerializeError::InvalidSignature);
    }
    offset += 4;

    let height = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
    offset += 4;

    let width = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
    offset += 4;

    let metadata = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
    offset += 4;

    let colorspace = ColorSpace::from_encoding((metadata >> 30 & 0b11) as u8)?;
    let variant = FractalVariant::from_encoding((metadata >> 28 & 0b11) as u8)?;

    let channel_data = deserialize_channel_data(&bytes, offset)?;

    Ok(CompressedImage {
        metadata: ImageMetadata {
            height,
            width,
            colorspace,
            variant,
        },
        channel_data,
    })
}

fn deserialize_channel_data(
    bytes: &Vec<u8>,
    mut offset: usize,
) -> Result<[Option<ChannelData>; 3], SerializeError> {
    let mut channel_data = [None, None, None];
    let mut ans_contexts: Vec<AnsContext> = vec![];
    let mut encoded_bytes: Vec<u8> = vec![];
    let mut value_prediction_parameters: Vec<[f32; 6]> = vec![[0.; 6]; 3];
    let mut width_prediction_parameters: Vec<[f32; 6]> = vec![[0.; 6]; 3];
    let mut i = 0;
    loop {
        match &bytes[offset..offset + 2] {
            Segments::PRD => {
                offset += 2;

                value_prediction_parameters[0] = bytes[offset..offset + 6 * 4]
                    .chunks_exact(4)
                    .map(|e| f32::from_le_bytes(e.try_into().unwrap()))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                offset += 6 * 4;

                value_prediction_parameters[1] = bytes[offset..offset + 6 * 4]
                    .chunks_exact(4)
                    .map(|e| f32::from_le_bytes(e.try_into().unwrap()))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                offset += 6 * 4;

                value_prediction_parameters[2] = bytes[offset..offset + 6 * 4]
                    .chunks_exact(4)
                    .map(|e| f32::from_le_bytes(e.try_into().unwrap()))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                offset += 6 * 4;

                width_prediction_parameters[0] = bytes[offset..offset + 6 * 4]
                    .chunks_exact(4)
                    .map(|e| f32::from_le_bytes(e.try_into().unwrap()))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                offset += 6 * 4;

                width_prediction_parameters[1] = bytes[offset..offset + 6 * 4]
                    .chunks_exact(4)
                    .map(|e| f32::from_le_bytes(e.try_into().unwrap()))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                offset += 6 * 4;

                width_prediction_parameters[2] = bytes[offset..offset + 6 * 4]
                    .chunks_exact(4)
                    .map(|e| f32::from_le_bytes(e.try_into().unwrap()))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                offset += 6 * 4;
            }
            Segments::EHD => {
                offset += 2;

                let max_freq_bits = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
                offset += 4;

                let off_distribution_len = usize::from_le_bytes(bytes[offset..offset + 8].try_into()?);
                offset += 8;

                let off_distribution_vals: Vec<u16> = bytes[offset..offset + off_distribution_len as usize * 2]
                    .chunks_exact(2)
                    .map(|e| u16::from_le_bytes(e.try_into().unwrap()))
                    .collect();

                offset += off_distribution_len as usize * 2;

                let mut context = AnsContext::new();

                context.max_freq_bits = max_freq_bits;
                context.off_distribution_values = off_distribution_vals;
                //context.freqs = (*freqs.into_boxed_slice()).try_into().unwrap();
                context.finalize_context(true, ans_contexts.len());
                ans_contexts.push(context)
            }
            Segments::DAT => {
                offset += 2;

                let data_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?) as usize;
                offset += 8;

                let data = bytes[offset..offset + data_len as usize].to_vec();
                offset += data_len;

                encoded_bytes = data;
            }
            Segments::EOC => {
                offset += 2;

                channel_data[i] = Some(ChannelData {
                    ans_contexts,
                    data: encoded_bytes,
                    value_prediction_parameters,
                    width_prediction_parameters,
                });
                value_prediction_parameters = vec![[0.; 6]; 3];
                width_prediction_parameters = vec![[0.; 6]; 3];
                ans_contexts = vec![];
                encoded_bytes = vec![];
                i += 1;
            }
            Segments::EOI => return Ok(channel_data),
            _other => return Err(SerializeError::MalformedImageBytes),
        }
    }
}
