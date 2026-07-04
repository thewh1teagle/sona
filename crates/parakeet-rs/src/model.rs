use std::path::Path;

use crate::backend::Backend;
use crate::encoder::EncoderTensor;
use crate::gguf::{Gguf, TensorInfo};
use crate::hparams::ParakeetHParams;
use crate::mel::{MelFrontend, MelSpectrogram};
use crate::tokenizer::Tokenizer;
use crate::weights::LoadedWeights;
use crate::{Error, Result};

pub struct ParakeetModel {
    gguf: Gguf,
    info: ModelInfo,
    hparams: ParakeetHParams,
    mel: MelFrontend,
    tokenizer: Tokenizer,
    weights: LoadedWeights,
    tensors: Vec<TensorInfo>,
}

#[derive(Debug, Clone)]
struct ModelInfo {
    architecture: String,
    name: String,
    variant: Option<String>,
    languages: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

impl ParakeetModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_backend(path, Backend::Auto)
    }

    pub fn load_with_backend(path: impl AsRef<Path>, backend: Backend) -> Result<Self> {
        let t_total = crate::timing::start();
        let t = crate::timing::start();
        let gguf = Gguf::open(path)?;
        crate::timing::log("load.gguf_open", t);
        let architecture = gguf.required_str("general.architecture")?;
        if architecture != "parakeet" {
            return Err(Error::UnsupportedArchitecture(architecture));
        }

        let name = gguf
            .optional_str("general.name")?
            .unwrap_or_else(|| "Parakeet".to_string());
        let variant = gguf.optional_str("stt.variant")?;
        if let Some(variant) = &variant {
            if variant != "tdt-0.6b-v3" && variant != "parakeet-tdt-0.6b-v3" {
                return Err(Error::UnsupportedVariant(variant.clone()));
            }
        }
        let languages = gguf.optional_str_array("general.languages")?;
        let t = crate::timing::start();
        let hparams = ParakeetHParams::read(&gguf)?;
        let mel = MelFrontend::from_hparams(&hparams)?;
        let tokenizer = Tokenizer::load(&gguf)?;
        let tensors = gguf.tensors();
        crate::timing::log("load.metadata", t);
        let t = crate::timing::start();
        let weights = LoadedWeights::load(gguf.path(), &hparams, backend)?;
        crate::timing::log("load.weights", t);
        crate::timing::log("load.total", t_total);

        Ok(Self {
            gguf,
            info: ModelInfo {
                architecture,
                name,
                variant,
                languages,
            },
            hparams,
            mel,
            tokenizer,
            weights,
            tensors,
        })
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<Transcript> {
        let t_total = crate::timing::start();
        if samples.is_empty() {
            return Err(Error::EmptyAudio);
        }

        let t = crate::timing::start();
        let mel = self.compute_mel(samples)?;
        crate::timing::log("transcribe.mel", t);
        let t = crate::timing::start();
        let enc = self.compute_encoder(&mel)?;
        crate::timing::log("transcribe.encoder", t);
        let t = crate::timing::start();
        let (raw_tokens, text) =
            crate::decoder::decode_tdt(&self.weights, &self.hparams, &enc, &self.tokenizer)?;
        crate::timing::log("transcribe.decoder", t);
        let frame_seconds = (self.hparams.enc_subsampling_factor as f32
            * self.hparams.fe_hop_length as f32)
            / self.hparams.fe_sample_rate as f32;
        let segments = if text.is_empty() {
            Vec::new()
        } else {
            let start = raw_tokens
                .first()
                .map(|token| token.step_at_emit as f32 * frame_seconds)
                .unwrap_or(0.0);
            let end = raw_tokens
                .last()
                .map(|token| (token.step_at_emit + token.duration_frames) as f32 * frame_seconds)
                .unwrap_or(start);
            vec![Segment {
                start,
                end,
                text: text.clone(),
            }]
        };
        crate::timing::log("transcribe.total", t_total);
        Ok(Transcript { text, segments })
    }

    pub fn compute_mel(&self, samples: &[f32]) -> Result<MelSpectrogram> {
        self.mel.compute(samples)
    }

    pub fn compute_pre_encode(&self, mel: &MelSpectrogram) -> Result<EncoderTensor> {
        crate::encoder::run_pre_encode(&self.weights, mel)
    }

    pub fn compute_encoder(&self, mel: &MelSpectrogram) -> Result<EncoderTensor> {
        crate::encoder::run_encoder(&self.weights, &self.hparams, mel)
    }

    pub fn name(&self) -> &str {
        &self.info.name
    }

    pub fn architecture(&self) -> &str {
        &self.info.architecture
    }

    pub fn variant(&self) -> Option<&str> {
        self.info.variant.as_deref()
    }

    pub fn languages(&self) -> &[String] {
        &self.info.languages
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn loaded_weight_blocks(&self) -> usize {
        self.weights.slots.blocks.len()
    }

    pub fn backend_name(&self) -> &str {
        &self.weights.backend_name
    }

    pub fn encoder_layers(&self) -> i32 {
        self.hparams.enc_n_layers
    }

    pub fn encoder_dim(&self) -> i32 {
        self.hparams.enc_d_model
    }

    pub fn sample_rate(&self) -> i32 {
        self.hparams.fe_sample_rate
    }

    pub fn frontend_sample_rate(&self) -> usize {
        self.mel.sample_rate()
    }

    pub fn model_path(&self) -> &Path {
        self.gguf.path()
    }
}
