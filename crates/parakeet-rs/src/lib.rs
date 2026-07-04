mod backend;
mod decoder;
mod encoder;
mod gguf;
mod hparams;
mod mel;
mod model;
mod timing;
mod tokenizer;
mod wav;
mod weights;

pub use encoder::EncoderTensor;
pub use mel::MelSpectrogram;
pub use model::{ParakeetModel, Segment, Transcript};
pub use wav::read_wav_16k_mono;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error("invalid string contains interior NUL byte")]
    Nul(#[from] std::ffi::NulError),
    #[error("failed to load GGUF model from {0}")]
    LoadGguf(std::path::PathBuf),
    #[error("missing GGUF key: {0}")]
    MissingKey(String),
    #[error("GGUF key {key} has unexpected type {actual}")]
    WrongKeyType { key: String, actual: String },
    #[error("invalid GGUF metadata: {0}")]
    InvalidGguf(String),
    #[error("invalid GGUF tensor catalog: {0}")]
    InvalidTensor(String),
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("unsupported model variant: {0}")]
    UnsupportedVariant(String),
    #[error("invalid WAV: expected 16 kHz mono PCM, got {sample_rate} Hz, {channels} channels, {bits_per_sample} bits")]
    InvalidWav {
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
    },
    #[error("empty audio")]
    EmptyAudio,
    #[error("Parakeet inference port is incomplete: {0}")]
    Incomplete(&'static str),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Hound(#[from] hound::Error),
}
pub use backend::Backend;
