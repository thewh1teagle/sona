use anyhow::{bail, Context as _};
use serde::Serialize;
use whisper_rs::{ContextOptions, Segment, StreamCallbacks, TranscribeOptions, TranscribeResult};

#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct EngineCapabilities {
    pub languages: Vec<String>,
    pub language_detection: bool,
    pub streaming: bool,
    pub translation: bool,
    pub timestamps: bool,
    pub text_prompts: bool,
}

pub enum Engine {
    Whisper(whisper_rs::Context),
    Nemotron(nemotron_rs::Model),
}

impl Engine {
    pub fn load(path: &str, options: ContextOptions) -> anyhow::Result<Self> {
        if path.ends_with(".gguf") {
            if let Ok(model) = nemotron_rs::Model::load(path) {
                return Ok(Self::Nemotron(model));
            }
        }
        Ok(Self::Whisper(whisper_rs::Context::new(path, options)?))
    }

    pub fn transcribe(&mut self, samples: &[f32], options: TranscribeOptions) -> anyhow::Result<TranscribeResult> {
        match self {
            Self::Whisper(context) => context.transcribe(samples, options).map_err(Into::into),
            Self::Nemotron(model) => {
                if options.translate {
                    bail!("Nemotron does not support translation");
                }
                if options.prompt.is_some() {
                    bail!("Nemotron does not support text prompts");
                }
                let language = if options.detect_language {
                    "auto"
                } else {
                    options.language.as_deref().unwrap_or("en-US")
                };
                let result = model.transcribe(samples, language).context("Nemotron inference failed")?;
                let start_cs = result.tokens.first().map_or(0, |token| token.frame as i64 * 8);
                let end_cs = result.tokens.last().map_or(0, |token| (token.frame as i64 + 1) * 8);
                Ok(TranscribeResult {
                    segments: if result.text.is_empty() {
                        Vec::new()
                    } else {
                        vec![Segment {
                            start: start_cs,
                            end: end_cs,
                            text: result.text,
                            no_speech_prob: 0.0,
                        }]
                    },
                })
            }
        }
    }

    pub fn transcribe_stream(
        &mut self,
        samples: &[f32],
        options: TranscribeOptions,
        mut callbacks: StreamCallbacks<'_>,
    ) -> anyhow::Result<TranscribeResult> {
        if let Self::Whisper(context) = self {
            return context.transcribe_stream(samples, options, callbacks).map_err(Into::into);
        }
        if callbacks.should_abort.as_mut().is_some_and(|callback| callback()) {
            bail!("transcription aborted");
        }
        if let Some(callback) = callbacks.on_progress.as_mut() {
            callback(0);
        }
        let result = self.transcribe(samples, options)?;
        for segment in &result.segments {
            if callbacks.should_abort.as_mut().is_some_and(|callback| callback()) {
                bail!("transcription aborted");
            }
            if let Some(callback) = callbacks.on_segment.as_mut() {
                callback(segment.clone());
            }
        }
        if let Some(callback) = callbacks.on_progress.as_mut() {
            callback(100);
        }
        Ok(result)
    }

    pub fn capabilities(&self) -> EngineCapabilities {
        match self {
            Self::Whisper(_) => EngineCapabilities {
                languages: whisper_rs::supported_languages(),
                language_detection: true,
                streaming: true,
                translation: true,
                timestamps: true,
                text_prompts: true,
            },
            Self::Nemotron(model) => EngineCapabilities {
                languages: model.info().languages.clone(),
                language_detection: true,
                streaming: false,
                translation: false,
                timestamps: true,
                text_prompts: false,
            },
        }
    }
}
