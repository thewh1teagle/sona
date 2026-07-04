use crate::gguf::Gguf;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeadKind {
    Tdt,
    Rnnt,
    Ctc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttContextStyle {
    Regular,
    ChunkedLimited,
    ChunkedLimitedWithRc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvNormType {
    BatchNorm,
    LayerNorm,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ParakeetHParams {
    pub(crate) head_kind: HeadKind,
    pub(crate) enc_n_layers: i32,
    pub(crate) enc_d_model: i32,
    pub(crate) enc_n_heads: i32,
    pub(crate) enc_d_ff: i32,
    pub(crate) enc_conv_kernel: i32,
    pub(crate) enc_subsampling_factor: i32,
    pub(crate) enc_subsampling_channels: i32,
    pub(crate) enc_pos_emb_max_len: i32,
    pub(crate) enc_use_bias: bool,
    pub(crate) enc_xscaling: bool,
    pub(crate) enc_att_context_left: i32,
    pub(crate) enc_att_context_right: i32,
    pub(crate) enc_att_context_size_choices: Vec<(i32, i32)>,
    pub(crate) enc_att_chunk_left_choices: Vec<i32>,
    pub(crate) enc_att_chunk_chunk_choices: Vec<i32>,
    pub(crate) enc_att_chunk_right_choices: Vec<i32>,
    pub(crate) enc_att_context_style: AttContextStyle,
    pub(crate) enc_conv_context_left: i32,
    pub(crate) enc_conv_context_right: i32,
    pub(crate) enc_conv_norm_type: ConvNormType,
    pub(crate) enc_stream_pre_encode_cache_size: i32,
    pub(crate) enc_stream_drop_extra_pre_encoded: i32,
    pub(crate) enc_stream_sampling_frames_first: i32,
    pub(crate) pred_hidden: i32,
    pub(crate) pred_n_layers: i32,
    pub(crate) pred_vocab: i32,
    pub(crate) joint_hidden: i32,
    pub(crate) joint_num_extra_outputs: i32,
    pub(crate) joint_activation: String,
    pub(crate) tdt_durations: Vec<i32>,
    pub(crate) tdt_max_symbols: i32,
    pub(crate) fe_type: String,
    pub(crate) fe_num_mels: i32,
    pub(crate) fe_sample_rate: i32,
    pub(crate) fe_n_fft: i32,
    pub(crate) fe_win_length: i32,
    pub(crate) fe_hop_length: i32,
    pub(crate) fe_window: String,
    pub(crate) fe_normalize: String,
    pub(crate) fe_dither: f32,
    pub(crate) fe_pre_emphasis: f32,
    pub(crate) fe_f_min: f32,
    pub(crate) fe_f_max: f32,
    pub(crate) has_prompt: bool,
    pub(crate) prompt_num_prompts: i32,
    pub(crate) prompt_hidden: i32,
    pub(crate) prompt_field: String,
    pub(crate) prompt_activation: String,
    pub(crate) prompt_dictionary_locales: Vec<String>,
    pub(crate) prompt_dictionary_indices: Vec<i32>,
    pub(crate) prompt_auto_id: i32,
}

impl ParakeetHParams {
    pub(crate) fn read(gguf: &Gguf) -> Result<Self> {
        let enc_n_layers = gguf.required_i32("stt.parakeet.encoder.n_layers")?;
        let enc_d_model = gguf.required_i32("stt.parakeet.encoder.d_model")?;
        let enc_n_heads = gguf.required_i32("stt.parakeet.encoder.n_heads")?;
        let enc_d_ff = gguf.required_i32("stt.parakeet.encoder.d_ff")?;
        let enc_conv_kernel = gguf.required_i32("stt.parakeet.encoder.conv_kernel")?;
        let enc_subsampling_factor =
            gguf.required_i32("stt.parakeet.encoder.subsampling_factor")?;
        let enc_subsampling_channels =
            gguf.required_i32("stt.parakeet.encoder.subsampling_channels")?;
        let enc_pos_emb_max_len = gguf.required_i32("stt.parakeet.encoder.pos_emb_max_len")?;
        let enc_use_bias = gguf.optional_bool("stt.parakeet.encoder.use_bias", false)?;
        let enc_xscaling = gguf.optional_bool("stt.parakeet.encoder.xscaling", false)?;
        let enc_att_context_left =
            gguf.optional_i32("stt.parakeet.encoder.att_context_left", -1)?;
        let enc_att_context_right =
            gguf.optional_i32("stt.parakeet.encoder.att_context_right", -1)?;

        let enc_att_context_size_choices =
            match gguf.optional_int_array("stt.parakeet.encoder.att_context_size_choices")? {
                Some(flat) => {
                    if flat.is_empty() || flat.len() % 2 != 0 {
                        return Err(Error::InvalidGguf(
                            "stt.parakeet.encoder.att_context_size_choices must be non-empty pairs"
                                .to_string(),
                        ));
                    }
                    let pairs = flat
                        .chunks_exact(2)
                        .map(|chunk| (chunk[0], chunk[1]))
                        .collect::<Vec<_>>();
                    if pairs[0] != (enc_att_context_left, enc_att_context_right) {
                        return Err(Error::InvalidGguf(
                            "att_context_size_choices[0] does not match att_context_left/right"
                                .to_string(),
                        ));
                    }
                    pairs
                }
                None => vec![(enc_att_context_left, enc_att_context_right)],
            };

        let enc_att_chunk_left_choices = gguf
            .optional_int_array("stt.parakeet.encoder.att_chunk_left_choices")?
            .unwrap_or_default();
        let enc_att_chunk_chunk_choices = gguf
            .optional_int_array("stt.parakeet.encoder.att_chunk_chunk_choices")?
            .unwrap_or_default();
        let enc_att_chunk_right_choices = gguf
            .optional_int_array("stt.parakeet.encoder.att_chunk_right_choices")?
            .unwrap_or_default();

        let style = gguf
            .optional_str("stt.parakeet.encoder.att_context_style")?
            .unwrap_or_else(|| "regular".to_string());
        let enc_att_context_style = match style.as_str() {
            "regular" => AttContextStyle::Regular,
            "chunked_limited" => AttContextStyle::ChunkedLimited,
            "chunked_limited_with_rc" => {
                if enc_att_chunk_left_choices.is_empty()
                    || enc_att_chunk_chunk_choices.is_empty()
                    || enc_att_chunk_right_choices.is_empty()
                {
                    return Err(Error::InvalidGguf(
                        "chunked_limited_with_rc requires non-empty L/C/R menus".to_string(),
                    ));
                }
                AttContextStyle::ChunkedLimitedWithRc
            }
            _ => {
                return Err(Error::InvalidGguf(format!(
                    "unsupported att_context_style {style}"
                )))
            }
        };

        let enc_conv_context_left =
            gguf.optional_i32("stt.parakeet.encoder.conv_context_left", -1)?;
        let enc_conv_context_right =
            gguf.optional_i32("stt.parakeet.encoder.conv_context_right", -1)?;
        let conv_norm_type = gguf
            .optional_str("stt.parakeet.encoder.conv_norm_type")?
            .unwrap_or_else(|| "batch_norm".to_string());
        let enc_conv_norm_type = match conv_norm_type.as_str() {
            "batch_norm" => ConvNormType::BatchNorm,
            "layer_norm" => ConvNormType::LayerNorm,
            _ => {
                return Err(Error::InvalidGguf(format!(
                    "unsupported conv_norm_type {conv_norm_type}"
                )))
            }
        };

        let enc_stream_pre_encode_cache_size =
            gguf.optional_i32("stt.parakeet.encoder.streaming.pre_encode_cache_size", 0)?;
        let enc_stream_drop_extra_pre_encoded =
            gguf.optional_i32("stt.parakeet.encoder.streaming.drop_extra_pre_encoded", 0)?;
        let enc_stream_sampling_frames_first =
            gguf.optional_i32("stt.parakeet.encoder.streaming.sampling_frames_first", 0)?;

        let head_kind_str = gguf
            .optional_str("stt.parakeet.head_kind")?
            .unwrap_or_else(|| "tdt".to_string());
        let head_kind = match head_kind_str.as_str() {
            "tdt" => HeadKind::Tdt,
            "rnnt" => HeadKind::Rnnt,
            "ctc" => HeadKind::Ctc,
            _ => {
                return Err(Error::InvalidGguf(format!(
                    "unsupported head_kind {head_kind_str}"
                )))
            }
        };

        let mut pred_hidden = 0;
        let mut pred_n_layers = 0;
        let mut pred_vocab = 0;
        let mut joint_hidden = 0;
        let mut joint_num_extra_outputs = 0;
        let mut joint_activation = String::new();
        if head_kind != HeadKind::Ctc {
            pred_hidden = gguf.required_i32("stt.parakeet.predictor.hidden")?;
            pred_n_layers = gguf.required_i32("stt.parakeet.predictor.n_layers")?;
            pred_vocab = gguf.required_i32("stt.parakeet.predictor.vocab")?;
            joint_hidden = gguf.required_i32("stt.parakeet.joint.hidden")?;
            joint_num_extra_outputs = gguf.required_i32("stt.parakeet.joint.num_extra_outputs")?;
            joint_activation = gguf.required_str("stt.parakeet.joint.activation")?;
        }

        let (tdt_durations, tdt_max_symbols) = match head_kind {
            HeadKind::Tdt => (
                gguf.optional_int_array("stt.parakeet.tdt.durations")?
                    .ok_or_else(|| Error::MissingKey("stt.parakeet.tdt.durations".to_string()))?,
                gguf.optional_i32("stt.parakeet.tdt.max_symbols", 10)?,
            ),
            HeadKind::Rnnt => (Vec::new(), 10),
            HeadKind::Ctc => (Vec::new(), 0),
        };

        let fe_type = gguf.required_str("stt.frontend.type")?;
        let fe_num_mels = gguf.required_i32("stt.frontend.num_mels")?;
        let fe_sample_rate = gguf.required_i32("stt.frontend.sample_rate")?;
        let fe_n_fft = gguf.required_i32("stt.frontend.n_fft")?;
        let fe_win_length = gguf.required_i32("stt.frontend.win_length")?;
        let fe_hop_length = gguf.required_i32("stt.frontend.hop_length")?;
        let fe_window = gguf.required_str("stt.frontend.window")?;
        let fe_normalize = gguf.required_str("stt.frontend.normalize")?;
        let fe_dither = gguf.required_f32("stt.frontend.dither")?;
        let fe_pre_emphasis = gguf.required_f32("stt.frontend.pre_emphasis")?;
        let fe_f_min = gguf.required_f32("stt.frontend.f_min")?;
        let fe_f_max = gguf.required_f32("stt.frontend.f_max")?;

        let mut hp = Self {
            head_kind,
            enc_n_layers,
            enc_d_model,
            enc_n_heads,
            enc_d_ff,
            enc_conv_kernel,
            enc_subsampling_factor,
            enc_subsampling_channels,
            enc_pos_emb_max_len,
            enc_use_bias,
            enc_xscaling,
            enc_att_context_left,
            enc_att_context_right,
            enc_att_context_size_choices,
            enc_att_chunk_left_choices,
            enc_att_chunk_chunk_choices,
            enc_att_chunk_right_choices,
            enc_att_context_style,
            enc_conv_context_left,
            enc_conv_context_right,
            enc_conv_norm_type,
            enc_stream_pre_encode_cache_size,
            enc_stream_drop_extra_pre_encoded,
            enc_stream_sampling_frames_first,
            pred_hidden,
            pred_n_layers,
            pred_vocab,
            joint_hidden,
            joint_num_extra_outputs,
            joint_activation,
            tdt_durations,
            tdt_max_symbols,
            fe_type,
            fe_num_mels,
            fe_sample_rate,
            fe_n_fft,
            fe_win_length,
            fe_hop_length,
            fe_window,
            fe_normalize,
            fe_dither,
            fe_pre_emphasis,
            fe_f_min,
            fe_f_max,
            has_prompt: false,
            prompt_num_prompts: 0,
            prompt_hidden: 0,
            prompt_field: String::new(),
            prompt_activation: String::new(),
            prompt_dictionary_locales: Vec::new(),
            prompt_dictionary_indices: Vec::new(),
            prompt_auto_id: -1,
        };

        if gguf.has_key("stt.parakeet.prompt.num_prompts")? {
            hp.prompt_num_prompts = gguf.required_i32("stt.parakeet.prompt.num_prompts")?;
            if hp.prompt_num_prompts > 0 {
                hp.has_prompt = true;
                hp.prompt_hidden = gguf.required_i32("stt.parakeet.prompt.hidden")?;
                hp.prompt_field = gguf.required_str("stt.parakeet.prompt.field")?;
                hp.prompt_activation = gguf.required_str("stt.parakeet.prompt.activation")?;
                hp.prompt_dictionary_locales =
                    gguf.optional_str_array("stt.parakeet.prompt.dictionary.locales")?;
                hp.prompt_dictionary_indices = gguf
                    .optional_int_array("stt.parakeet.prompt.dictionary.indices")?
                    .ok_or_else(|| {
                        Error::MissingKey("stt.parakeet.prompt.dictionary.indices".to_string())
                    })?;
                hp.prompt_auto_id = if gguf.has_key("stt.parakeet.prompt.auto_id")? {
                    gguf.required_i32("stt.parakeet.prompt.auto_id")?
                } else {
                    -1
                };
            }
        }

        hp.validate()?;
        Ok(hp)
    }

    fn validate(&self) -> Result<()> {
        if self.enc_n_layers <= 0
            || self.enc_d_model <= 0
            || self.enc_n_heads <= 0
            || self.enc_d_ff <= 0
            || self.enc_conv_kernel <= 0
            || self.enc_subsampling_factor <= 0
            || self.enc_subsampling_channels <= 0
        {
            return Err(Error::InvalidGguf(
                "encoder hparams must be positive".to_string(),
            ));
        }
        if self.enc_d_model % self.enc_n_heads != 0 {
            return Err(Error::InvalidGguf(
                "encoder d_model must be divisible by n_heads".to_string(),
            ));
        }
        if self.head_kind != HeadKind::Ctc {
            if self.pred_hidden <= 0 || self.pred_n_layers <= 0 || self.pred_vocab <= 1 {
                return Err(Error::InvalidGguf(
                    "predictor hparams must be positive".to_string(),
                ));
            }
            if self.joint_hidden <= 0 || self.joint_num_extra_outputs < 0 {
                return Err(Error::InvalidGguf("joint hparams invalid".to_string()));
            }
            if !matches!(self.joint_activation.as_str(), "relu" | "sigmoid" | "tanh") {
                return Err(Error::InvalidGguf(format!(
                    "unsupported joint activation {}",
                    self.joint_activation
                )));
            }
        }
        if self.head_kind == HeadKind::Tdt {
            if self.tdt_durations.is_empty() {
                return Err(Error::InvalidGguf("TDT durations are empty".to_string()));
            }
            if self.tdt_durations.len() != self.joint_num_extra_outputs as usize {
                return Err(Error::InvalidGguf(
                    "TDT durations length must equal joint.num_extra_outputs".to_string(),
                ));
            }
            if self.tdt_durations.iter().any(|&d| d < 0) || self.tdt_max_symbols < 0 {
                return Err(Error::InvalidGguf("invalid TDT durations/cap".to_string()));
            }
        }
        if self.fe_num_mels <= 0
            || self.fe_sample_rate <= 0
            || self.fe_n_fft <= 0
            || self.fe_win_length <= 0
            || self.fe_hop_length <= 0
        {
            return Err(Error::InvalidGguf(
                "frontend dimensions must be positive".to_string(),
            ));
        }
        if self.fe_win_length > self.fe_n_fft {
            return Err(Error::InvalidGguf("win_length > n_fft".to_string()));
        }
        if self.fe_f_min < 0.0 || self.fe_f_max <= self.fe_f_min || self.fe_dither < 0.0 {
            return Err(Error::InvalidGguf("invalid frontend scalar".to_string()));
        }
        if self.fe_type != "mel" || self.fe_window != "hann" {
            return Err(Error::InvalidGguf(
                "unsupported frontend type/window".to_string(),
            ));
        }
        if self.fe_normalize != "per_feature" && self.fe_normalize != "none" {
            return Err(Error::InvalidGguf(
                "unsupported frontend normalization".to_string(),
            ));
        }
        if self.fe_n_fft & (self.fe_n_fft - 1) != 0 {
            return Err(Error::InvalidGguf(
                "n_fft must be a power of two".to_string(),
            ));
        }
        if self.has_prompt {
            if self.prompt_dictionary_locales.len() != self.prompt_dictionary_indices.len() {
                return Err(Error::InvalidGguf(
                    "prompt dictionary locales/indices length mismatch".to_string(),
                ));
            }
            if self.prompt_activation != "relu" || self.prompt_hidden <= 0 {
                return Err(Error::InvalidGguf(
                    "invalid prompt MLP metadata".to_string(),
                ));
            }
        }
        Ok(())
    }
}
