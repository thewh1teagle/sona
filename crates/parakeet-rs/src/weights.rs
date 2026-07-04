use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::ptr;

use whisper_rs::ffi;

use crate::backend::{backend_name, Backend};
use crate::hparams::{ConvNormType, HeadKind, ParakeetHParams};
use crate::{Error, Result};

pub(crate) struct LoadedWeights {
    pub(crate) ctx_meta: *mut ffi::ggml_context,
    pub(crate) backend: ffi::ggml_backend_t,
    pub(crate) buffer: ffi::ggml_backend_buffer_t,
    pub(crate) slots: ParakeetWeights,
    pub(crate) backend_name: String,
    _path: PathBuf,
}

impl LoadedWeights {
    pub(crate) fn load(
        path: impl AsRef<Path>,
        hparams: &ParakeetHParams,
        backend_kind: Backend,
    ) -> Result<Self> {
        let path = path.as_ref();
        let c_path = CString::new(path.to_string_lossy().as_bytes())?;
        let mut ctx_meta: *mut ffi::ggml_context = ptr::null_mut();
        let params = ffi::gguf_init_params {
            no_alloc: true,
            ctx: &mut ctx_meta,
        };
        let gguf_data = unsafe { ffi::gguf_init_from_file(c_path.as_ptr(), params) };
        if gguf_data.is_null() || ctx_meta.is_null() {
            if !gguf_data.is_null() {
                unsafe { ffi::gguf_free(gguf_data) };
            }
            return Err(Error::LoadGguf(path.to_path_buf()));
        }

        let load_result = (|| {
            let slots = ParakeetWeights::build(ctx_meta, hparams)?;
            let backend = backend_kind.init()?;
            let backend_name = backend_name(backend);
            let buffer = unsafe { ffi::ggml_backend_alloc_ctx_tensors(ctx_meta, backend) };
            if buffer.is_null() {
                unsafe { ffi::ggml_backend_free(backend) };
                return Err(Error::InvalidTensor(
                    "ggml_backend_alloc_ctx_tensors failed".to_string(),
                ));
            }
            unsafe {
                ffi::ggml_backend_buffer_set_usage(
                    buffer,
                    ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
                );
            }
            stream_tensor_data(path, gguf_data, ctx_meta)?;
            Ok(Self {
                ctx_meta,
                backend,
                buffer,
                slots,
                backend_name,
                _path: path.to_path_buf(),
            })
        })();

        unsafe { ffi::gguf_free(gguf_data) };
        match load_result {
            Ok(loaded) => Ok(loaded),
            Err(err) => {
                unsafe { ffi::ggml_free(ctx_meta) };
                Err(err)
            }
        }
    }
}

impl Drop for LoadedWeights {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                ffi::ggml_backend_buffer_free(self.buffer);
            }
            if !self.backend.is_null() {
                ffi::ggml_backend_free(self.backend);
            }
            if !self.ctx_meta.is_null() {
                ffi::ggml_free(self.ctx_meta);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct TensorSlot(pub(crate) *mut ffi::ggml_tensor);

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct ParakeetWeights {
    pub(crate) pre_encode: PreEncodeWeights,
    pub(crate) blocks: Vec<BlockWeights>,
    pub(crate) predictor: Option<PredictorWeights>,
    pub(crate) joint: Option<JointWeights>,
    pub(crate) prompt: Option<PromptWeights>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct PreEncodeWeights {
    pub(crate) conv0_w: TensorSlot,
    pub(crate) conv0_b: TensorSlot,
    pub(crate) conv2_w: TensorSlot,
    pub(crate) conv2_b: TensorSlot,
    pub(crate) conv3_w: TensorSlot,
    pub(crate) conv3_b: TensorSlot,
    pub(crate) conv5_w: TensorSlot,
    pub(crate) conv5_b: TensorSlot,
    pub(crate) conv6_w: TensorSlot,
    pub(crate) conv6_b: TensorSlot,
    pub(crate) out_w: TensorSlot,
    pub(crate) out_b: TensorSlot,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct BlockWeights {
    pub(crate) norm_ff1_w: TensorSlot,
    pub(crate) norm_ff1_b: TensorSlot,
    pub(crate) ff1_lin1_w: TensorSlot,
    pub(crate) ff1_lin1_b: Option<TensorSlot>,
    pub(crate) ff1_lin2_w: TensorSlot,
    pub(crate) ff1_lin2_b: Option<TensorSlot>,
    pub(crate) norm_attn_w: TensorSlot,
    pub(crate) norm_attn_b: TensorSlot,
    pub(crate) attn_q_w: TensorSlot,
    pub(crate) attn_q_b: Option<TensorSlot>,
    pub(crate) attn_k_w: TensorSlot,
    pub(crate) attn_k_b: Option<TensorSlot>,
    pub(crate) attn_v_w: TensorSlot,
    pub(crate) attn_v_b: Option<TensorSlot>,
    pub(crate) attn_out_w: TensorSlot,
    pub(crate) attn_out_b: Option<TensorSlot>,
    pub(crate) attn_pos_w: TensorSlot,
    pub(crate) attn_pos_u: TensorSlot,
    pub(crate) attn_pos_v: TensorSlot,
    pub(crate) norm_conv_w: TensorSlot,
    pub(crate) norm_conv_b: TensorSlot,
    pub(crate) conv_pw1_w: TensorSlot,
    pub(crate) conv_pw1_b: Option<TensorSlot>,
    pub(crate) conv_dw_w: TensorSlot,
    pub(crate) conv_dw_b: Option<TensorSlot>,
    pub(crate) conv_pw2_w: TensorSlot,
    pub(crate) conv_pw2_b: Option<TensorSlot>,
    pub(crate) conv_bn_w: TensorSlot,
    pub(crate) conv_bn_b: TensorSlot,
    pub(crate) conv_bn_rm: Option<TensorSlot>,
    pub(crate) conv_bn_rv: Option<TensorSlot>,
    pub(crate) norm_ff2_w: TensorSlot,
    pub(crate) norm_ff2_b: TensorSlot,
    pub(crate) ff2_lin1_w: TensorSlot,
    pub(crate) ff2_lin1_b: Option<TensorSlot>,
    pub(crate) ff2_lin2_w: TensorSlot,
    pub(crate) ff2_lin2_b: Option<TensorSlot>,
    pub(crate) norm_out_w: TensorSlot,
    pub(crate) norm_out_b: TensorSlot,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct PredictorWeights {
    pub(crate) embed_w: TensorSlot,
    pub(crate) lstm: Vec<LstmWeights>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct LstmWeights {
    pub(crate) wx: TensorSlot,
    pub(crate) wh: TensorSlot,
    pub(crate) b: TensorSlot,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct JointWeights {
    pub(crate) enc_w: TensorSlot,
    pub(crate) enc_b: TensorSlot,
    pub(crate) pred_w: TensorSlot,
    pub(crate) pred_b: TensorSlot,
    pub(crate) out_w: TensorSlot,
    pub(crate) out_b: TensorSlot,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct PromptWeights {
    pub(crate) mlp0_w: TensorSlot,
    pub(crate) mlp0_b: TensorSlot,
    pub(crate) mlp2_w: TensorSlot,
    pub(crate) mlp2_b: TensorSlot,
}

impl ParakeetWeights {
    fn build(ctx: *mut ffi::ggml_context, hp: &ParakeetHParams) -> Result<Self> {
        let channels = i64::from(hp.enc_subsampling_channels);
        let d_model = i64::from(hp.enc_d_model);
        let d_ff = i64::from(hp.enc_d_ff);
        let n_heads = i64::from(hp.enc_n_heads);
        let head_dim = d_model / n_heads;
        let k = i64::from(hp.enc_conv_kernel);
        let pred_h = i64::from(hp.pred_hidden);
        let pred_v = i64::from(hp.pred_vocab);
        let joint_h = i64::from(hp.joint_hidden);
        let joint_n = i64::from((hp.pred_vocab - 1) + hp.joint_num_extra_outputs + 1);
        let pre_encode_in = channels * i64::from(pre_encode_f_prime(hp));

        let pre_encode = PreEncodeWeights {
            conv0_w: get_conv(ctx, "enc.pre_encode.conv.0.weight", &[3, 3, 1, channels])?,
            conv0_b: get_f32(ctx, "enc.pre_encode.conv.0.bias", &[channels])?,
            conv2_w: get_conv(ctx, "enc.pre_encode.conv.2.weight", &[3, 3, 1, channels])?,
            conv2_b: get_f32(ctx, "enc.pre_encode.conv.2.bias", &[channels])?,
            conv3_w: get_conv(
                ctx,
                "enc.pre_encode.conv.3.weight",
                &[1, 1, channels, channels],
            )?,
            conv3_b: get_f32(ctx, "enc.pre_encode.conv.3.bias", &[channels])?,
            conv5_w: get_conv(ctx, "enc.pre_encode.conv.5.weight", &[3, 3, 1, channels])?,
            conv5_b: get_f32(ctx, "enc.pre_encode.conv.5.bias", &[channels])?,
            conv6_w: get_conv(
                ctx,
                "enc.pre_encode.conv.6.weight",
                &[1, 1, channels, channels],
            )?,
            conv6_b: get_f32(ctx, "enc.pre_encode.conv.6.bias", &[channels])?,
            out_w: get_lin(ctx, "enc.pre_encode.out.weight", &[pre_encode_in, d_model])?,
            out_b: get_f32(ctx, "enc.pre_encode.out.bias", &[d_model])?,
        };

        let mut blocks = Vec::with_capacity(hp.enc_n_layers as usize);
        for i in 0..hp.enc_n_layers {
            blocks.push(BlockWeights {
                norm_ff1_w: get_f32(ctx, &lname("enc.blocks.%d.norm_ff1.weight", i), &[d_model])?,
                norm_ff1_b: get_f32(ctx, &lname("enc.blocks.%d.norm_ff1.bias", i), &[d_model])?,
                ff1_lin1_w: get_lin(
                    ctx,
                    &lname("enc.blocks.%d.ff1.linear1.weight", i),
                    &[d_model, d_ff],
                )?,
                ff1_lin1_b: hp
                    .enc_use_bias
                    .then(|| get_f32(ctx, &lname("enc.blocks.%d.ff1.linear1.bias", i), &[d_ff]))
                    .transpose()?,
                ff1_lin2_w: get_lin(
                    ctx,
                    &lname("enc.blocks.%d.ff1.linear2.weight", i),
                    &[d_ff, d_model],
                )?,
                ff1_lin2_b: hp
                    .enc_use_bias
                    .then(|| get_f32(ctx, &lname("enc.blocks.%d.ff1.linear2.bias", i), &[d_model]))
                    .transpose()?,
                norm_attn_w: get_f32(ctx, &lname("enc.blocks.%d.norm_attn.weight", i), &[d_model])?,
                norm_attn_b: get_f32(ctx, &lname("enc.blocks.%d.norm_attn.bias", i), &[d_model])?,
                attn_q_w: get_lin(
                    ctx,
                    &lname("enc.blocks.%d.attn.linear_q.weight", i),
                    &[d_model, d_model],
                )?,
                attn_q_b: hp
                    .enc_use_bias
                    .then(|| {
                        get_f32(
                            ctx,
                            &lname("enc.blocks.%d.attn.linear_q.bias", i),
                            &[d_model],
                        )
                    })
                    .transpose()?,
                attn_k_w: get_lin(
                    ctx,
                    &lname("enc.blocks.%d.attn.linear_k.weight", i),
                    &[d_model, d_model],
                )?,
                attn_k_b: hp
                    .enc_use_bias
                    .then(|| {
                        get_f32(
                            ctx,
                            &lname("enc.blocks.%d.attn.linear_k.bias", i),
                            &[d_model],
                        )
                    })
                    .transpose()?,
                attn_v_w: get_lin(
                    ctx,
                    &lname("enc.blocks.%d.attn.linear_v.weight", i),
                    &[d_model, d_model],
                )?,
                attn_v_b: hp
                    .enc_use_bias
                    .then(|| {
                        get_f32(
                            ctx,
                            &lname("enc.blocks.%d.attn.linear_v.bias", i),
                            &[d_model],
                        )
                    })
                    .transpose()?,
                attn_out_w: get_lin(
                    ctx,
                    &lname("enc.blocks.%d.attn.linear_out.weight", i),
                    &[d_model, d_model],
                )?,
                attn_out_b: hp
                    .enc_use_bias
                    .then(|| {
                        get_f32(
                            ctx,
                            &lname("enc.blocks.%d.attn.linear_out.bias", i),
                            &[d_model],
                        )
                    })
                    .transpose()?,
                attn_pos_w: get_lin(
                    ctx,
                    &lname("enc.blocks.%d.attn.linear_pos.weight", i),
                    &[d_model, d_model],
                )?,
                attn_pos_u: get_f32(
                    ctx,
                    &lname("enc.blocks.%d.attn.pos_bias_u", i),
                    &[head_dim, n_heads],
                )?,
                attn_pos_v: get_f32(
                    ctx,
                    &lname("enc.blocks.%d.attn.pos_bias_v", i),
                    &[head_dim, n_heads],
                )?,
                norm_conv_w: get_f32(ctx, &lname("enc.blocks.%d.norm_conv.weight", i), &[d_model])?,
                norm_conv_b: get_f32(ctx, &lname("enc.blocks.%d.norm_conv.bias", i), &[d_model])?,
                conv_pw1_w: get_conv(
                    ctx,
                    &lname("enc.blocks.%d.conv.pointwise1.weight", i),
                    &[1, d_model, 2 * d_model],
                )?,
                conv_pw1_b: hp
                    .enc_use_bias
                    .then(|| {
                        get_f32(
                            ctx,
                            &lname("enc.blocks.%d.conv.pointwise1.bias", i),
                            &[2 * d_model],
                        )
                    })
                    .transpose()?,
                conv_dw_w: get_conv(
                    ctx,
                    &lname("enc.blocks.%d.conv.depthwise.weight", i),
                    &[k, 1, d_model],
                )?,
                conv_dw_b: hp
                    .enc_use_bias
                    .then(|| {
                        get_f32(
                            ctx,
                            &lname("enc.blocks.%d.conv.depthwise.bias", i),
                            &[d_model],
                        )
                    })
                    .transpose()?,
                conv_pw2_w: get_conv(
                    ctx,
                    &lname("enc.blocks.%d.conv.pointwise2.weight", i),
                    &[1, d_model, d_model],
                )?,
                conv_pw2_b: hp
                    .enc_use_bias
                    .then(|| {
                        get_f32(
                            ctx,
                            &lname("enc.blocks.%d.conv.pointwise2.bias", i),
                            &[d_model],
                        )
                    })
                    .transpose()?,
                conv_bn_w: get_f32(ctx, &lname("enc.blocks.%d.conv.bn.weight", i), &[d_model])?,
                conv_bn_b: get_f32(ctx, &lname("enc.blocks.%d.conv.bn.bias", i), &[d_model])?,
                conv_bn_rm: (hp.enc_conv_norm_type == ConvNormType::BatchNorm)
                    .then(|| {
                        get_f32(
                            ctx,
                            &lname("enc.blocks.%d.conv.bn.running_mean", i),
                            &[d_model],
                        )
                    })
                    .transpose()?,
                conv_bn_rv: (hp.enc_conv_norm_type == ConvNormType::BatchNorm)
                    .then(|| {
                        get_f32(
                            ctx,
                            &lname("enc.blocks.%d.conv.bn.running_var", i),
                            &[d_model],
                        )
                    })
                    .transpose()?,
                norm_ff2_w: get_f32(ctx, &lname("enc.blocks.%d.norm_ff2.weight", i), &[d_model])?,
                norm_ff2_b: get_f32(ctx, &lname("enc.blocks.%d.norm_ff2.bias", i), &[d_model])?,
                ff2_lin1_w: get_lin(
                    ctx,
                    &lname("enc.blocks.%d.ff2.linear1.weight", i),
                    &[d_model, d_ff],
                )?,
                ff2_lin1_b: hp
                    .enc_use_bias
                    .then(|| get_f32(ctx, &lname("enc.blocks.%d.ff2.linear1.bias", i), &[d_ff]))
                    .transpose()?,
                ff2_lin2_w: get_lin(
                    ctx,
                    &lname("enc.blocks.%d.ff2.linear2.weight", i),
                    &[d_ff, d_model],
                )?,
                ff2_lin2_b: hp
                    .enc_use_bias
                    .then(|| get_f32(ctx, &lname("enc.blocks.%d.ff2.linear2.bias", i), &[d_model]))
                    .transpose()?,
                norm_out_w: get_f32(ctx, &lname("enc.blocks.%d.norm_out.weight", i), &[d_model])?,
                norm_out_b: get_f32(ctx, &lname("enc.blocks.%d.norm_out.bias", i), &[d_model])?,
            });
        }

        if hp.head_kind == HeadKind::Ctc {
            return Err(Error::UnsupportedVariant(
                "CTC Parakeet is not wired in parakeet-rs yet".to_string(),
            ));
        }

        let mut lstm = Vec::with_capacity(hp.pred_n_layers as usize);
        for i in 0..hp.pred_n_layers {
            let gates = 4 * pred_h;
            lstm.push(LstmWeights {
                wx: get_lin(ctx, &lname("pred.lstm.%d.Wx", i), &[pred_h, gates])?,
                wh: get_lin(ctx, &lname("pred.lstm.%d.Wh", i), &[pred_h, gates])?,
                b: get_f32(ctx, &lname("pred.lstm.%d.bias", i), &[gates])?,
            });
        }
        let predictor = Some(PredictorWeights {
            embed_w: get_lin(ctx, "pred.embed.weight", &[pred_h, pred_v])?,
            lstm,
        });

        let joint = Some(JointWeights {
            enc_w: get_lin(ctx, "joint.enc.weight", &[d_model, joint_h])?,
            enc_b: get_f32(ctx, "joint.enc.bias", &[joint_h])?,
            pred_w: get_lin(ctx, "joint.pred.weight", &[pred_h, joint_h])?,
            pred_b: get_f32(ctx, "joint.pred.bias", &[joint_h])?,
            out_w: get_lin(ctx, "joint.out.weight", &[joint_h, joint_n])?,
            out_b: get_f32(ctx, "joint.out.bias", &[joint_n])?,
        });

        let prompt = hp
            .has_prompt
            .then(|| {
                let prompt_h = i64::from(hp.prompt_hidden);
                let in_dim = d_model + i64::from(hp.prompt_num_prompts);
                Ok::<PromptWeights, Error>(PromptWeights {
                    mlp0_w: get_lin(ctx, "prompt.mlp.0.weight", &[in_dim, prompt_h])?,
                    mlp0_b: get_f32(ctx, "prompt.mlp.0.bias", &[prompt_h])?,
                    mlp2_w: get_lin(ctx, "prompt.mlp.2.weight", &[prompt_h, d_model])?,
                    mlp2_b: get_f32(ctx, "prompt.mlp.2.bias", &[d_model])?,
                })
            })
            .transpose()?;

        Ok(Self {
            pre_encode,
            blocks,
            predictor,
            joint,
            prompt,
        })
    }
}

fn pre_encode_f_prime(hp: &ParakeetHParams) -> i32 {
    let total_pad = 2;
    let mut dim = hp.fe_num_mels;
    for _ in 0..3 {
        dim = ((dim + total_pad - 3) / 2) + 1;
    }
    dim
}

fn get_f32(ctx: *mut ffi::ggml_context, name: &str, expected: &[i64]) -> Result<TensorSlot> {
    find_tensor(ctx, name, &[ffi::ggml_type_GGML_TYPE_F32], expected)
}

fn get_conv(ctx: *mut ffi::ggml_context, name: &str, expected: &[i64]) -> Result<TensorSlot> {
    find_tensor(
        ctx,
        name,
        &[ffi::ggml_type_GGML_TYPE_F32, ffi::ggml_type_GGML_TYPE_F16],
        expected,
    )
}

fn get_lin(ctx: *mut ffi::ggml_context, name: &str, expected: &[i64]) -> Result<TensorSlot> {
    find_tensor(
        ctx,
        name,
        &[
            ffi::ggml_type_GGML_TYPE_F32,
            ffi::ggml_type_GGML_TYPE_F16,
            ffi::ggml_type_GGML_TYPE_BF16,
            ffi::ggml_type_GGML_TYPE_Q4_0,
            ffi::ggml_type_GGML_TYPE_Q4_1,
            ffi::ggml_type_GGML_TYPE_Q5_0,
            ffi::ggml_type_GGML_TYPE_Q5_1,
            ffi::ggml_type_GGML_TYPE_Q8_0,
            ffi::ggml_type_GGML_TYPE_Q4_K,
            ffi::ggml_type_GGML_TYPE_Q5_K,
            ffi::ggml_type_GGML_TYPE_Q6_K,
        ],
        expected,
    )
}

fn find_tensor(
    ctx: *mut ffi::ggml_context,
    name: &str,
    allowed_types: &[ffi::ggml_type],
    expected_ne: &[i64],
) -> Result<TensorSlot> {
    let c_name = CString::new(name)?;
    let tensor = unsafe { ffi::ggml_get_tensor(ctx, c_name.as_ptr()) };
    if tensor.is_null() {
        return Err(Error::InvalidTensor(format!("missing tensor {name}")));
    }
    let actual_type = unsafe { (*tensor).type_ };
    if !allowed_types.contains(&actual_type) {
        return Err(Error::InvalidTensor(format!(
            "tensor {name} type mismatch: got {}",
            ggml_type_name(actual_type)
        )));
    }
    if expected_ne.is_empty() || expected_ne.len() > 4 {
        return Err(Error::InvalidTensor(format!(
            "bad expected shape for tensor {name}"
        )));
    }
    let ne = unsafe { (*tensor).ne };
    for (i, expected) in expected_ne.iter().enumerate() {
        if ne[i] != *expected {
            return Err(Error::InvalidTensor(format!(
                "tensor {name} shape mismatch: expected ne[{i}]={expected}, got {}",
                ne[i]
            )));
        }
    }
    for (i, actual) in ne.iter().enumerate().skip(expected_ne.len()) {
        if *actual != 1 {
            return Err(Error::InvalidTensor(format!(
                "tensor {name} has unexpected non-1 ne[{i}]={actual}"
            )));
        }
    }
    Ok(TensorSlot(tensor))
}

fn stream_tensor_data(
    path: &Path,
    gguf_data: *const ffi::gguf_context,
    ctx_meta: *mut ffi::ggml_context,
) -> Result<()> {
    let mut file = File::open(path)?;
    let data_offset = unsafe { ffi::gguf_get_data_offset(gguf_data) };
    let mut staging = Vec::new();
    let mut tensor = unsafe { ffi::ggml_get_first_tensor(ctx_meta) };
    while !tensor.is_null() {
        let name = tensor_name(tensor);
        let c_name = CString::new(name.as_str())?;
        let idx = unsafe { ffi::gguf_find_tensor(gguf_data, c_name.as_ptr()) };
        if idx < 0 {
            return Err(Error::InvalidTensor(format!(
                "tensor {name} not found in GGUF data"
            )));
        }
        let tensor_offset = unsafe { ffi::gguf_get_tensor_offset(gguf_data, idx) };
        let nbytes = unsafe { ffi::ggml_nbytes(tensor) };
        file.seek(SeekFrom::Start((data_offset + tensor_offset) as u64))?;
        staging.resize(nbytes, 0);
        file.read_exact(&mut staging)?;
        unsafe {
            ffi::ggml_backend_tensor_set(tensor, staging.as_ptr().cast(), 0, nbytes);
            tensor = ffi::ggml_get_next_tensor(ctx_meta, tensor);
        }
    }
    Ok(())
}

fn tensor_name(tensor: *const ffi::ggml_tensor) -> String {
    unsafe { CStr::from_ptr((*tensor).name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn ggml_type_name(ty: ffi::ggml_type) -> String {
    unsafe { CStr::from_ptr(ffi::ggml_type_name(ty)) }
        .to_string_lossy()
        .into_owned()
}

fn lname(fmt: &str, layer_idx: i32) -> String {
    fmt.replace("%d", &layer_idx.to_string())
}
