use std::path::Path;

use crate::{Error, Result};

pub fn read_wav_16k_mono(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000
        || spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        return Err(Error::InvalidWav {
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            bits_per_sample: spec.bits_per_sample,
        });
    }

    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.map(|v| f32::from(v) / f32::from(i16::MAX)))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if samples.is_empty() {
        return Err(Error::EmptyAudio);
    }
    Ok(samples)
}
