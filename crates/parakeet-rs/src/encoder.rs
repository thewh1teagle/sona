use std::ffi::CString;
use std::ptr;

use whisper_rs::ffi;

use crate::hparams::ParakeetHParams;
use crate::mel::MelSpectrogram;
use crate::weights::{BlockWeights, LoadedWeights, PreEncodeWeights, TensorSlot};
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct EncoderTensor {
    pub values: Vec<f32>,
    pub d_model: usize,
    pub n_frames: usize,
}

struct BnFusedInput {
    scale_tensor: *mut ffi::ggml_tensor,
    bias_tensor: *mut ffi::ggml_tensor,
    scale: Vec<f32>,
    bias: Vec<f32>,
}

struct SigmoidOnesInput {
    tensor: *mut ffi::ggml_tensor,
    values: Vec<f32>,
}

struct ComputeContext {
    ctx: *mut ffi::ggml_context,
}

impl ComputeContext {
    fn new(mem_size: usize) -> Result<Self> {
        let params = ffi::ggml_init_params {
            mem_size,
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = unsafe { ffi::ggml_init(params) };
        if ctx.is_null() {
            return Err(Error::InvalidTensor(
                "ggml_init compute context failed".to_string(),
            ));
        }
        Ok(Self { ctx })
    }
}

impl Drop for ComputeContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { ffi::ggml_free(self.ctx) };
        }
    }
}

struct ComputeBuffer {
    buffer: ffi::ggml_backend_buffer_t,
}

impl ComputeBuffer {
    fn new(ctx: *mut ffi::ggml_context, backend: ffi::ggml_backend_t) -> Result<Self> {
        let buffer = unsafe { ffi::ggml_backend_alloc_ctx_tensors(ctx, backend) };
        if buffer.is_null() {
            return Err(Error::InvalidTensor(
                "ggml_backend_alloc_ctx_tensors for compute graph failed".to_string(),
            ));
        }
        Ok(Self { buffer })
    }
}

impl Drop for ComputeBuffer {
    fn drop(&mut self) {
        if !self.buffer.is_null() {
            unsafe { ffi::ggml_backend_buffer_free(self.buffer) };
        }
    }
}

pub(crate) fn run_pre_encode(
    weights: &LoadedWeights,
    mel: &MelSpectrogram,
) -> Result<EncoderTensor> {
    let t_total = crate::timing::start();
    let compute = ComputeContext::new(8 * 1024 * 1024)?;
    let pe = &weights.slots.pre_encode;
    let mel_in = unsafe {
        ffi::ggml_new_tensor_4d(
            compute.ctx,
            ffi::ggml_type_GGML_TYPE_F32,
            mel.n_frames as i64,
            mel.n_mels as i64,
            1,
            1,
        )
    };
    ensure_tensor(mel_in, "mel.in")?;
    set_name(mel_in, "mel.in")?;
    unsafe { ffi::ggml_set_input(mel_in) };

    let direct_dw = !weights.backend_name.to_ascii_lowercase().contains("metal");
    let t = crate::timing::start();
    let out = unsafe { build_pre_encode(compute.ctx, pe, mel_in, direct_dw)? };
    crate::timing::log("pre_encode.build_graph", t);
    unsafe { ffi::ggml_set_output(out) };
    let graph = unsafe { ffi::ggml_new_graph_custom(compute.ctx, 1024, false) };
    if graph.is_null() {
        return Err(Error::InvalidTensor(
            "ggml_new_graph_custom failed".to_string(),
        ));
    }
    unsafe { ffi::ggml_build_forward_expand(graph, out) };

    let t = crate::timing::start();
    let _compute_buffer = ComputeBuffer::new(compute.ctx, weights.backend)?;
    crate::timing::log("pre_encode.alloc", t);
    unsafe {
        let t = crate::timing::start();
        ffi::ggml_backend_tensor_set(
            mel_in,
            mel.values.as_ptr().cast(),
            0,
            mel.values.len() * std::mem::size_of::<f32>(),
        );
        let status = ffi::ggml_backend_graph_compute(weights.backend, graph);
        crate::timing::log("pre_encode.compute", t);
        if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
            return Err(Error::InvalidTensor(format!(
                "pre-encode graph compute failed with status {status}"
            )));
        }
    }

    let d_model = unsafe { (*out).ne[0] as usize };
    let n_frames = unsafe { (*out).ne[1] as usize };
    let mut values = vec![0.0_f32; d_model * n_frames];
    unsafe {
        let t = crate::timing::start();
        ffi::ggml_backend_tensor_get(
            out,
            values.as_mut_ptr().cast(),
            0,
            values.len() * std::mem::size_of::<f32>(),
        );
        crate::timing::log("pre_encode.readback", t);
    }
    crate::timing::log("pre_encode.total", t_total);
    Ok(EncoderTensor {
        values,
        d_model,
        n_frames,
    })
}

pub(crate) fn run_encoder(
    weights: &LoadedWeights,
    hparams: &ParakeetHParams,
    mel: &MelSpectrogram,
) -> Result<EncoderTensor> {
    let t_total = crate::timing::start();
    let compute = ComputeContext::new(16 * 1024 * 1024)?;
    let mel_in = unsafe {
        ffi::ggml_new_tensor_4d(
            compute.ctx,
            ffi::ggml_type_GGML_TYPE_F32,
            mel.n_frames as i64,
            mel.n_mels as i64,
            1,
            1,
        )
    };
    ensure_tensor(mel_in, "mel.in")?;
    set_name(mel_in, "mel.in")?;
    unsafe { ffi::ggml_set_input(mel_in) };

    let direct_dw = !weights.backend_name.to_ascii_lowercase().contains("metal");
    let t = crate::timing::start();
    let mut x =
        unsafe { build_pre_encode(compute.ctx, &weights.slots.pre_encode, mel_in, direct_dw)? };
    crate::timing::log("encoder.build_pre_encode", t);
    let t_enc = unsafe { (*x).ne[1] };
    let pos_len = 2 * t_enc - 1;
    let pos_emb = unsafe {
        ffi::ggml_new_tensor_2d(
            compute.ctx,
            ffi::ggml_type_GGML_TYPE_F32,
            i64::from(hparams.enc_d_model),
            pos_len,
        )
    };
    ensure_tensor(pos_emb, "pos_emb.in")?;
    set_name(pos_emb, "pos_emb.in")?;
    unsafe { ffi::ggml_set_input(pos_emb) };

    let t = crate::timing::start();
    let bn_inputs = build_bn_inputs(compute.ctx, weights, hparams)?;
    crate::timing::log("encoder.build_bn_inputs", t);
    let use_sigmoid_compat = weights.backend_name.to_ascii_lowercase().contains("metal");
    let mut sigmoid_ones = Vec::with_capacity(weights.slots.blocks.len());
    for (i, block) in weights.slots.blocks.iter().enumerate() {
        let t = crate::timing::start();
        x = unsafe {
            build_conformer_block(
                compute.ctx,
                x,
                pos_emb,
                block,
                hparams,
                &bn_inputs[i],
                &mut sigmoid_ones,
                use_sigmoid_compat,
                direct_dw,
            )?
        };
        if crate::timing::enabled() && (i == 0 || i == 11 || i == weights.slots.blocks.len() - 1) {
            crate::timing::log(&format!("encoder.build_block.{i}"), t);
        }
    }

    if hparams.has_prompt {
        let prompt = weights
            .slots
            .prompt
            .as_ref()
            .ok_or_else(|| Error::InvalidTensor("prompt weights missing".to_string()))?;
        let one_hot = unsafe {
            ffi::ggml_new_tensor_4d(
                compute.ctx,
                ffi::ggml_type_GGML_TYPE_F32,
                i64::from(hparams.prompt_num_prompts),
                t_enc,
                1,
                1,
            )
        };
        ensure_tensor(one_hot, "prompt.one_hot.in")?;
        set_name(one_hot, "prompt.one_hot.in")?;
        unsafe { ffi::ggml_set_input(one_hot) };
        unsafe {
            let cat = ffi::ggml_concat(compute.ctx, x, one_hot, 0);
            let mut h = ffi::ggml_mul_mat(compute.ctx, ptr_of(prompt.mlp0_w), cat);
            h = ffi::ggml_add(compute.ctx, h, ptr_of(prompt.mlp0_b));
            h = ffi::ggml_relu(compute.ctx, h);
            x = ffi::ggml_mul_mat(compute.ctx, ptr_of(prompt.mlp2_w), h);
            x = ffi::ggml_add(compute.ctx, x, ptr_of(prompt.mlp2_b));
        }
    }

    unsafe { ffi::ggml_set_output(x) };
    let graph = unsafe { ffi::ggml_new_graph_custom(compute.ctx, 8192, false) };
    if graph.is_null() {
        return Err(Error::InvalidTensor(
            "ggml_new_graph_custom failed".to_string(),
        ));
    }
    unsafe { ffi::ggml_build_forward_expand(graph, x) };

    let t = crate::timing::start();
    let _compute_buffer = ComputeBuffer::new(compute.ctx, weights.backend)?;
    crate::timing::log("encoder.alloc", t);
    let pos_buf = build_pos_emb(hparams.enc_d_model as usize, pos_len as usize);
    let prompt_buf = hparams
        .has_prompt
        .then(|| build_prompt_one_hot(hparams, t_enc as usize))
        .transpose()?;
    unsafe {
        let t_upload = crate::timing::start();
        ffi::ggml_backend_tensor_set(
            mel_in,
            mel.values.as_ptr().cast(),
            0,
            mel.values.len() * std::mem::size_of::<f32>(),
        );
        ffi::ggml_backend_tensor_set(
            pos_emb,
            pos_buf.as_ptr().cast(),
            0,
            pos_buf.len() * std::mem::size_of::<f32>(),
        );
        if hparams.has_prompt {
            let one_hot = find_input_by_name(compute.ctx, "prompt.one_hot.in")?;
            let prompt_buf = prompt_buf.as_ref().expect("prompt buffer exists");
            ffi::ggml_backend_tensor_set(
                one_hot,
                prompt_buf.as_ptr().cast(),
                0,
                prompt_buf.len() * std::mem::size_of::<f32>(),
            );
        }
        for bn in &bn_inputs {
            ffi::ggml_backend_tensor_set(
                bn.scale_tensor,
                bn.scale.as_ptr().cast(),
                0,
                bn.scale.len() * std::mem::size_of::<f32>(),
            );
            ffi::ggml_backend_tensor_set(
                bn.bias_tensor,
                bn.bias.as_ptr().cast(),
                0,
                bn.bias.len() * std::mem::size_of::<f32>(),
            );
        }
        for ones in &sigmoid_ones {
            ffi::ggml_backend_tensor_set(
                ones.tensor,
                ones.values.as_ptr().cast(),
                0,
                ones.values.len() * std::mem::size_of::<f32>(),
            );
        }
        crate::timing::log("encoder.upload_inputs", t_upload);
        let t_compute = crate::timing::start();
        let status = ffi::ggml_backend_graph_compute(weights.backend, graph);
        crate::timing::log("encoder.compute", t_compute);
        if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
            return Err(Error::InvalidTensor(format!(
                "encoder graph compute failed with status {status}"
            )));
        }
    }

    let d_model = unsafe { (*x).ne[0] as usize };
    let n_frames = unsafe { (*x).ne[1] as usize };
    let mut values = vec![0.0_f32; d_model * n_frames];
    unsafe {
        let t = crate::timing::start();
        ffi::ggml_backend_tensor_get(
            x,
            values.as_mut_ptr().cast(),
            0,
            values.len() * std::mem::size_of::<f32>(),
        );
        crate::timing::log("encoder.readback", t);
    }
    crate::timing::log("encoder.total", t_total);
    Ok(EncoderTensor {
        values,
        d_model,
        n_frames,
    })
}

unsafe fn build_pre_encode(
    ctx: *mut ffi::ggml_context,
    pe: &PreEncodeWeights,
    mel_in: *mut ffi::ggml_tensor,
    direct_dw: bool,
) -> Result<*mut ffi::ggml_tensor> {
    let mut x = ffi::ggml_permute(ctx, mel_in, 1, 0, 2, 3);
    x = ffi::ggml_cont(ctx, x);

    x = ffi::ggml_conv_2d(ctx, ptr_of(pe.conv0_w), x, 2, 2, 1, 1, 1, 1);
    x = add_conv_bias(ctx, x, pe.conv0_b);
    x = ffi::ggml_relu(ctx, x);

    x = if direct_dw {
        ffi::ggml_conv_2d_dw_direct(
            ctx,
            dw_kernel_for_direct(ctx, ptr_of(pe.conv2_w)),
            x,
            2,
            2,
            1,
            1,
            1,
            1,
        )
    } else {
        conv_2d_dw_f32(ctx, ptr_of(pe.conv2_w), x, 2, 2, 1, 1, 1, 1)
    };
    x = add_conv_bias(ctx, x, pe.conv2_b);

    x = ffi::ggml_conv_2d(ctx, ptr_of(pe.conv3_w), x, 1, 1, 0, 0, 1, 1);
    x = add_conv_bias(ctx, x, pe.conv3_b);
    x = ffi::ggml_relu(ctx, x);

    x = if direct_dw {
        ffi::ggml_conv_2d_dw_direct(
            ctx,
            dw_kernel_for_direct(ctx, ptr_of(pe.conv5_w)),
            x,
            2,
            2,
            1,
            1,
            1,
            1,
        )
    } else {
        conv_2d_dw_f32(ctx, ptr_of(pe.conv5_w), x, 2, 2, 1, 1, 1, 1)
    };
    x = add_conv_bias(ctx, x, pe.conv5_b);

    x = ffi::ggml_conv_2d(ctx, ptr_of(pe.conv6_w), x, 1, 1, 0, 0, 1, 1);
    x = add_conv_bias(ctx, x, pe.conv6_b);
    x = ffi::ggml_relu(ctx, x);

    let f_prime = (*x).ne[0];
    let t_enc = (*x).ne[1];
    let channels = (*x).ne[2];
    let batch = (*x).ne[3];
    let pre_encode_in = f_prime * channels;
    if pre_encode_in != (*ptr_of(pe.out_w)).ne[0] {
        return Err(Error::InvalidTensor(format!(
            "pre_encode_in mismatch: got {pre_encode_in}, out_w expects {}",
            (*ptr_of(pe.out_w)).ne[0]
        )));
    }

    x = ffi::ggml_permute(ctx, x, 0, 2, 1, 3);
    x = ffi::ggml_cont(ctx, x);
    x = ffi::ggml_reshape_3d(ctx, x, pre_encode_in, t_enc, batch);
    x = ffi::ggml_mul_mat(ctx, ptr_of(pe.out_w), x);
    let d_model = (*ptr_of(pe.out_b)).ne[0];
    let bias = ffi::ggml_reshape_4d(ctx, ptr_of(pe.out_b), d_model, 1, 1, 1);
    x = ffi::ggml_add(ctx, x, bias);
    ensure_tensor(x, "enc.pre_encode.out")?;
    set_name(x, "enc.pre_encode.out")?;
    Ok(x)
}

unsafe fn build_conformer_block(
    ctx: *mut ffi::ggml_context,
    mut x: *mut ffi::ggml_tensor,
    pos_emb: *mut ffi::ggml_tensor,
    block: &BlockWeights,
    hp: &ParakeetHParams,
    bn: &BnFusedInput,
    sigmoid_ones: &mut Vec<SigmoidOnesInput>,
    use_sigmoid_compat: bool,
    direct_dw: bool,
) -> Result<*mut ffi::ggml_tensor> {
    x = macaron_ff_residual(
        ctx,
        x,
        block.norm_ff1_w,
        block.norm_ff1_b,
        block.ff1_lin1_w,
        block.ff1_lin1_b,
        block.ff1_lin2_w,
        block.ff1_lin2_b,
    );

    let x_norm = layer_norm(ctx, x, block.norm_attn_w, Some(block.norm_attn_b));
    let attn = rel_pos_mhsa(ctx, x_norm, pos_emb, block, hp);
    x = ffi::ggml_add(ctx, x, attn);

    let x_norm = layer_norm(ctx, x, block.norm_conv_w, Some(block.norm_conv_b));
    let conv = conv_module(
        ctx,
        x_norm,
        block,
        hp,
        bn,
        sigmoid_ones,
        use_sigmoid_compat,
        direct_dw,
    );
    x = ffi::ggml_add(ctx, x, conv);

    x = macaron_ff_residual(
        ctx,
        x,
        block.norm_ff2_w,
        block.norm_ff2_b,
        block.ff2_lin1_w,
        block.ff2_lin1_b,
        block.ff2_lin2_w,
        block.ff2_lin2_b,
    );

    Ok(layer_norm(ctx, x, block.norm_out_w, Some(block.norm_out_b)))
}

unsafe fn layer_norm(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    gamma: TensorSlot,
    beta: Option<TensorSlot>,
) -> *mut ffi::ggml_tensor {
    let mut y = ffi::ggml_norm(ctx, x, 1.0e-5);
    y = ffi::ggml_mul(ctx, y, ptr_of(gamma));
    if let Some(beta) = beta {
        y = ffi::ggml_add(ctx, y, ptr_of(beta));
    }
    y
}

unsafe fn feed_forward(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    lin1_w: TensorSlot,
    lin1_b: Option<TensorSlot>,
    lin2_w: TensorSlot,
    lin2_b: Option<TensorSlot>,
) -> *mut ffi::ggml_tensor {
    let mut y = ffi::ggml_mul_mat(ctx, ptr_of(lin1_w), x);
    if (*ptr_of(lin1_w)).type_ == ffi::ggml_type_GGML_TYPE_F16 {
        ffi::ggml_mul_mat_set_prec(y, ffi::ggml_prec_GGML_PREC_F32);
    }
    if let Some(bias) = lin1_b {
        y = ffi::ggml_add(ctx, y, ptr_of(bias));
    }
    y = ffi::ggml_silu(ctx, y);
    y = ffi::ggml_mul_mat(ctx, ptr_of(lin2_w), y);
    if (*ptr_of(lin2_w)).type_ == ffi::ggml_type_GGML_TYPE_F16 {
        ffi::ggml_mul_mat_set_prec(y, ffi::ggml_prec_GGML_PREC_F32);
    }
    if let Some(bias) = lin2_b {
        y = ffi::ggml_add(ctx, y, ptr_of(bias));
    }
    y
}

unsafe fn macaron_ff_residual(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    norm_w: TensorSlot,
    norm_b: TensorSlot,
    lin1_w: TensorSlot,
    lin1_b: Option<TensorSlot>,
    lin2_w: TensorSlot,
    lin2_b: Option<TensorSlot>,
) -> *mut ffi::ggml_tensor {
    let mut y = layer_norm(ctx, x, norm_w, Some(norm_b));
    y = feed_forward(ctx, y, lin1_w, lin1_b, lin2_w, lin2_b);
    y = ffi::ggml_scale(ctx, y, 0.5);
    ffi::ggml_add(ctx, x, y)
}

unsafe fn rel_pos_mhsa(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    pos_emb: *mut ffi::ggml_tensor,
    block: &BlockWeights,
    hp: &ParakeetHParams,
) -> *mut ffi::ggml_tensor {
    let d_model = i64::from(hp.enc_d_model);
    let n_head = i64::from(hp.enc_n_heads);
    let head_dim = d_model / n_head;
    let t = (*x).ne[1];
    let batch = (*x).ne[2];
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut q = ffi::ggml_mul_mat(ctx, ptr_of(block.attn_q_w), x);
    if let Some(bias) = block.attn_q_b {
        q = ffi::ggml_add(ctx, q, ptr_of(bias));
    }
    let mut k = ffi::ggml_mul_mat(ctx, ptr_of(block.attn_k_w), x);
    if let Some(bias) = block.attn_k_b {
        k = ffi::ggml_add(ctx, k, ptr_of(bias));
    }
    let mut v = ffi::ggml_mul_mat(ctx, ptr_of(block.attn_v_w), x);
    if let Some(bias) = block.attn_v_b {
        v = ffi::ggml_add(ctx, v, ptr_of(bias));
    }
    let mut p = ffi::ggml_mul_mat(ctx, ptr_of(block.attn_pos_w), pos_emb);

    q = ffi::ggml_reshape_4d(ctx, q, head_dim, n_head, t, batch);
    let mut q_u = ffi::ggml_add(ctx, q, ptr_of(block.attn_pos_u));
    let mut q_v = ffi::ggml_add(ctx, q, ptr_of(block.attn_pos_v));
    q_u = ffi::ggml_permute(ctx, q_u, 0, 2, 1, 3);
    q_v = ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, q_v, 0, 2, 1, 3));

    k = ffi::ggml_reshape_4d(ctx, k, head_dim, n_head, t, batch);
    k = ffi::ggml_permute(ctx, k, 0, 2, 1, 3);

    v = ffi::ggml_reshape_4d(ctx, v, head_dim, n_head, t, batch);
    v = ffi::ggml_permute(ctx, v, 0, 2, 1, 3);

    p = ffi::ggml_reshape_4d(ctx, p, head_dim, n_head, (*pos_emb).ne[1], 1);
    p = ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, p, 0, 2, 1, 3));

    let mut matrix_bd = ffi::ggml_mul_mat(ctx, p, q_v);
    matrix_bd = rel_shift(ctx, matrix_bd);
    matrix_bd = ffi::ggml_view_4d(
        ctx,
        matrix_bd,
        t,
        t,
        n_head,
        batch,
        (*matrix_bd).nb[1],
        (*matrix_bd).nb[2],
        (*matrix_bd).nb[3],
        0,
    );

    let use_flash = std::env::var_os("PARAKEET_RS_NO_FLASH").is_none();
    let mut o = if use_flash {
        matrix_bd = ffi::ggml_cont(ctx, matrix_bd);
        matrix_bd = ffi::ggml_scale(ctx, matrix_bd, scale);
        matrix_bd = ffi::ggml_cast(ctx, matrix_bd, ffi::ggml_type_GGML_TYPE_F16);

        if (*ptr_of(block.attn_k_w)).type_ != ffi::ggml_type_GGML_TYPE_F32 {
            k = ffi::ggml_cast(ctx, k, ffi::ggml_type_GGML_TYPE_F16);
            v = ffi::ggml_cast(ctx, v, ffi::ggml_type_GGML_TYPE_F16);
        }

        ffi::ggml_flash_attn_ext(ctx, q_u, k, v, matrix_bd, scale, 0.0, 0.0)
    } else {
        let mut kq = ffi::ggml_mul_mat(ctx, k, q_u);
        kq = ffi::ggml_add(ctx, kq, matrix_bd);
        let kq_soft = ffi::ggml_soft_max_ext(ctx, kq, ptr::null_mut(), scale, 0.0);
        let v_t = ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, v, 1, 0, 2, 3));
        let mut manual_o = ffi::ggml_mul_mat(ctx, v_t, kq_soft);
        manual_o = ffi::ggml_permute(ctx, manual_o, 0, 2, 1, 3);
        ffi::ggml_cont(ctx, manual_o)
    };
    o = ffi::ggml_reshape_3d(ctx, o, d_model, t, batch);
    o = ffi::ggml_mul_mat(ctx, ptr_of(block.attn_out_w), o);
    if let Some(bias) = block.attn_out_b {
        o = ffi::ggml_add(ctx, o, ptr_of(bias));
    }
    o
}

unsafe fn rel_shift(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
) -> *mut ffi::ggml_tensor {
    let pos_len = (*x).ne[0];
    let t = (*x).ne[1];
    let heads = (*x).ne[2];
    let batch = (*x).ne[3];
    let zero_template =
        ffi::ggml_new_tensor_4d(ctx, ffi::ggml_type_GGML_TYPE_F32, 1, t, heads, batch);
    let zeros = ffi::ggml_fill(ctx, zero_template, 0.0);
    let mut y = ffi::ggml_concat(ctx, zeros, x, 0);
    y = ffi::ggml_reshape_4d(ctx, y, t, pos_len + 1, heads, batch);
    y = ffi::ggml_view_4d(
        ctx,
        y,
        t,
        pos_len,
        heads,
        batch,
        (*y).nb[1],
        (*y).nb[2],
        (*y).nb[3],
        (*y).nb[1],
    );
    y = ffi::ggml_cont(ctx, y);
    ffi::ggml_reshape_4d(ctx, y, pos_len, t, heads, batch)
}

unsafe fn conv_module(
    ctx: *mut ffi::ggml_context,
    mut x: *mut ffi::ggml_tensor,
    block: &BlockWeights,
    hp: &ParakeetHParams,
    bn: &BnFusedInput,
    sigmoid_ones: &mut Vec<SigmoidOnesInput>,
    use_sigmoid_compat: bool,
    direct_dw: bool,
) -> *mut ffi::ggml_tensor {
    let d_model = i64::from(hp.enc_d_model);
    let t = (*x).ne[1];
    let batch = (*x).ne[2];

    let pw1 = ffi::ggml_reshape_2d(ctx, ptr_of(block.conv_pw1_w), d_model, 2 * d_model);
    x = ffi::ggml_mul_mat(ctx, pw1, x);
    if (*ptr_of(block.conv_pw1_w)).type_ == ffi::ggml_type_GGML_TYPE_F16 {
        ffi::ggml_mul_mat_set_prec(x, ffi::ggml_prec_GGML_PREC_F32);
    }
    if let Some(bias) = block.conv_pw1_b {
        x = ffi::ggml_add(ctx, x, ptr_of(bias));
    }

    let half = (*x).ne[0] / 2;
    let gate = ffi::ggml_view_3d(ctx, x, half, t, batch, (*x).nb[1], (*x).nb[2], 0);
    let value = ffi::ggml_view_3d(
        ctx,
        x,
        half,
        t,
        batch,
        (*x).nb[1],
        (*x).nb[2],
        half as usize * ffi::ggml_element_size(x),
    );
    let sigmoid_value = if use_sigmoid_compat {
        sigmoid_compat(ctx, value, sigmoid_ones)
    } else {
        ffi::ggml_sigmoid(ctx, value)
    };
    x = ffi::ggml_mul(ctx, gate, sigmoid_value);
    x = ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, x, 1, 0, 2, 3));

    x = if direct_dw {
        conv_1d_dw_direct(ctx, ptr_of(block.conv_dw_w), x, hp.enc_conv_kernel, d_model)
    } else {
        conv_1d_dw_f32(
            ctx,
            ptr_of(block.conv_dw_w),
            x,
            1,
            (hp.enc_conv_kernel - 1) / 2,
            1,
        )
    };
    if let Some(bias) = block.conv_dw_b {
        let bias_r = ffi::ggml_reshape_2d(ctx, ptr_of(bias), 1, d_model);
        x = ffi::ggml_add(ctx, x, bias_r);
    }
    x = fused_batch_norm(ctx, x, bn);
    x = ffi::ggml_silu(ctx, x);

    x = ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, x, 1, 0, 2, 3));
    let pw2 = ffi::ggml_reshape_2d(ctx, ptr_of(block.conv_pw2_w), d_model, d_model);
    x = ffi::ggml_mul_mat(ctx, pw2, x);
    if (*ptr_of(block.conv_pw2_w)).type_ == ffi::ggml_type_GGML_TYPE_F16 {
        ffi::ggml_mul_mat_set_prec(x, ffi::ggml_prec_GGML_PREC_F32);
    }
    if let Some(bias) = block.conv_pw2_b {
        x = ffi::ggml_add(ctx, x, ptr_of(bias));
    }
    x
}

unsafe fn fused_batch_norm(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    bn: &BnFusedInput,
) -> *mut ffi::ggml_tensor {
    let d_model = (*bn.scale_tensor).ne[0];
    let scale = ffi::ggml_reshape_2d(ctx, bn.scale_tensor, 1, d_model);
    let bias = ffi::ggml_reshape_2d(ctx, bn.bias_tensor, 1, d_model);
    let y = ffi::ggml_mul(ctx, x, scale);
    ffi::ggml_add(ctx, y, bias)
}

unsafe fn conv_1d_dw_f32(
    ctx: *mut ffi::ggml_context,
    kernel: *mut ffi::ggml_tensor,
    data: *mut ffi::ggml_tensor,
    stride: i32,
    padding: i32,
    dilation: i32,
) -> *mut ffi::ggml_tensor {
    let data_4d = ffi::ggml_reshape_4d(ctx, data, (*data).ne[0], 1, (*data).ne[1], (*data).ne[2]);
    let im2col = ffi::ggml_im2col(
        ctx,
        kernel,
        data_4d,
        stride,
        0,
        padding,
        0,
        dilation,
        0,
        false,
        (*kernel).type_,
    );
    let result = ffi::ggml_mul_mat(ctx, im2col, kernel);
    ffi::ggml_reshape_3d(ctx, result, (*result).ne[0], (*result).ne[2], 1)
}

fn build_bn_inputs(
    ctx: *mut ffi::ggml_context,
    weights: &LoadedWeights,
    hparams: &ParakeetHParams,
) -> Result<Vec<BnFusedInput>> {
    let d = hparams.enc_d_model as usize;
    let mut out = Vec::with_capacity(weights.slots.blocks.len());
    for (i, block) in weights.slots.blocks.iter().enumerate() {
        let rm = block.conv_bn_rm.ok_or_else(|| {
            Error::InvalidTensor(format!("block {i} batch norm running_mean missing"))
        })?;
        let rv = block.conv_bn_rv.ok_or_else(|| {
            Error::InvalidTensor(format!("block {i} batch norm running_var missing"))
        })?;

        let mut bn_w = read_f32_tensor(weights, block.conv_bn_w, d)?;
        let bn_b = read_f32_tensor(weights, block.conv_bn_b, d)?;
        let rm = read_f32_tensor(weights, rm, d)?;
        let rv = read_f32_tensor(weights, rv, d)?;
        let mut fused_b = vec![0.0_f32; d];
        for c in 0..d {
            let scale = bn_w[c] / (rv[c] + 1.0e-5).sqrt();
            bn_w[c] = scale;
            fused_b[c] = bn_b[c] - rm[c] * scale;
        }

        let scale_tensor = unsafe {
            ffi::ggml_new_tensor_1d(
                ctx,
                ffi::ggml_type_GGML_TYPE_F32,
                hparams.enc_d_model as i64,
            )
        };
        let bias_tensor = unsafe {
            ffi::ggml_new_tensor_1d(
                ctx,
                ffi::ggml_type_GGML_TYPE_F32,
                hparams.enc_d_model as i64,
            )
        };
        ensure_tensor(scale_tensor, "conv.bn.fused_scale")?;
        ensure_tensor(bias_tensor, "conv.bn.fused_bias")?;
        set_name(scale_tensor, &format!("enc.blocks.{i}.conv.bn.fused_scale"))?;
        set_name(bias_tensor, &format!("enc.blocks.{i}.conv.bn.fused_bias"))?;
        unsafe {
            ffi::ggml_set_input(scale_tensor);
            ffi::ggml_set_input(bias_tensor);
        }
        out.push(BnFusedInput {
            scale_tensor,
            bias_tensor,
            scale: bn_w,
            bias: fused_b,
        });
    }
    Ok(out)
}

fn read_f32_tensor(weights: &LoadedWeights, slot: TensorSlot, len: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0_f32; len];
    unsafe {
        ffi::ggml_backend_tensor_get(
            ptr_of(slot),
            out.as_mut_ptr().cast(),
            0,
            out.len() * std::mem::size_of::<f32>(),
        );
    }
    let _ = weights;
    Ok(out)
}

fn build_pos_emb(d_model: usize, pos_len: usize) -> Vec<f32> {
    let zero_index = (pos_len - 1) / 2;
    let ln_10000 = 10000.0_f32.ln();
    let div_terms = (0..d_model / 2)
        .map(|k| ((2 * k) as f32 * (-ln_10000 / d_model as f32)).exp())
        .collect::<Vec<_>>();
    let mut out = vec![0.0_f32; pos_len * d_model];
    for i in 0..pos_len {
        let pos = (zero_index as isize - i as isize) as f32;
        let row = &mut out[i * d_model..(i + 1) * d_model];
        for k in 0..d_model / 2 {
            let x = pos * div_terms[k];
            row[2 * k] = x.sin();
            row[2 * k + 1] = x.cos();
        }
    }
    out
}

fn build_prompt_one_hot(hparams: &ParakeetHParams, t_enc: usize) -> Result<Vec<f32>> {
    let prompt_count = hparams.prompt_num_prompts as usize;
    let prompt_id = if hparams.prompt_auto_id >= 0 {
        hparams.prompt_auto_id
    } else {
        *hparams
            .prompt_dictionary_indices
            .first()
            .ok_or_else(|| Error::InvalidGguf("prompt dictionary is empty".to_string()))?
    };
    if prompt_id < 0 || prompt_id as usize >= prompt_count {
        return Err(Error::InvalidGguf(format!(
            "prompt id {prompt_id} out of range [0, {prompt_count})"
        )));
    }
    let mut out = vec![0.0_f32; prompt_count * t_enc];
    for t in 0..t_enc {
        out[t * prompt_count + prompt_id as usize] = 1.0;
    }
    Ok(out)
}

unsafe fn find_input_by_name(
    ctx: *mut ffi::ggml_context,
    name: &str,
) -> Result<*mut ffi::ggml_tensor> {
    let c_name = CString::new(name)?;
    let tensor = ffi::ggml_get_tensor(ctx, c_name.as_ptr());
    ensure_tensor(tensor, name)?;
    Ok(tensor)
}

unsafe fn sigmoid_compat(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    sigmoid_ones: &mut Vec<SigmoidOnesInput>,
) -> *mut ffi::ggml_tensor {
    let x = ffi::ggml_cont(ctx, x);
    let ones = ffi::ggml_new_tensor_3d(
        ctx,
        ffi::ggml_type_GGML_TYPE_F32,
        (*x).ne[0],
        (*x).ne[1],
        (*x).ne[2],
    );
    ffi::ggml_set_input(ones);
    let name = format!("sigmoid.ones.{}", sigmoid_ones.len());
    let _ = set_name(ones, &name);
    let n = ((*x).ne[0] * (*x).ne[1] * (*x).ne[2]) as usize;
    sigmoid_ones.push(SigmoidOnesInput {
        tensor: ones,
        values: vec![1.0; n],
    });
    let neg = ffi::ggml_scale(ctx, x, -1.0);
    let denom = ffi::ggml_add(ctx, ones, ffi::ggml_exp(ctx, neg));
    ffi::ggml_div(ctx, ones, denom)
}

unsafe fn conv_2d_dw_f32(
    ctx: *mut ffi::ggml_context,
    kernel: *mut ffi::ggml_tensor,
    data: *mut ffi::ggml_tensor,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    d0: i32,
    d1: i32,
) -> *mut ffi::ggml_tensor {
    let new_a = ffi::ggml_reshape_4d(
        ctx,
        kernel,
        (*kernel).ne[0],
        (*kernel).ne[1],
        1,
        (*kernel).ne[2] * (*kernel).ne[3],
    );
    let data_4d = ffi::ggml_reshape_4d(
        ctx,
        data,
        (*data).ne[0],
        (*data).ne[1],
        1,
        (*data).ne[2] * (*data).ne[3],
    );
    let im2col = ffi::ggml_im2col(
        ctx,
        new_a,
        data_4d,
        s0,
        s1,
        p0,
        p1,
        d0,
        d1,
        true,
        (*kernel).type_,
    );
    let new_b = ffi::ggml_reshape_4d(
        ctx,
        im2col,
        (*im2col).ne[0],
        (*im2col).ne[2] * (*im2col).ne[1],
        (*data).ne[2],
        (*data).ne[3],
    );
    let new_a = ffi::ggml_reshape_4d(
        ctx,
        new_a,
        (*new_a).ne[0] * (*new_a).ne[1],
        (*new_a).ne[2],
        (*new_a).ne[3],
        1,
    );
    let result = ffi::ggml_mul_mat(ctx, new_a, new_b);
    ffi::ggml_reshape_4d(
        ctx,
        result,
        (*im2col).ne[1],
        (*im2col).ne[2],
        (*data).ne[2],
        (*data).ne[3],
    )
}

unsafe fn dw_kernel_for_direct(
    ctx: *mut ffi::ggml_context,
    kernel: *mut ffi::ggml_tensor,
) -> *mut ffi::ggml_tensor {
    if (*kernel).type_ == ffi::ggml_type_GGML_TYPE_F32 {
        kernel
    } else {
        ffi::ggml_cast(ctx, kernel, ffi::ggml_type_GGML_TYPE_F32)
    }
}

unsafe fn conv_1d_dw_direct(
    ctx: *mut ffi::ggml_context,
    kernel: *mut ffi::ggml_tensor,
    data: *mut ffi::ggml_tensor,
    conv_kernel: i32,
    d_model: i64,
) -> *mut ffi::ggml_tensor {
    let batch = (*data).ne[2];
    let knl = ffi::ggml_reshape_4d(
        ctx,
        dw_kernel_for_direct(ctx, kernel),
        i64::from(conv_kernel),
        1,
        1,
        d_model,
    );
    let data_4d = ffi::ggml_reshape_4d(ctx, data, (*data).ne[0], 1, (*data).ne[1], batch);
    let out = ffi::ggml_conv_2d_dw_direct(ctx, knl, data_4d, 1, 1, (conv_kernel - 1) / 2, 0, 1, 1);
    ffi::ggml_reshape_3d(ctx, out, (*out).ne[0], (*out).ne[2], (*out).ne[3])
}

unsafe fn add_conv_bias(
    ctx: *mut ffi::ggml_context,
    conv_out: *mut ffi::ggml_tensor,
    bias: TensorSlot,
) -> *mut ffi::ggml_tensor {
    let channels = (*ptr_of(bias)).ne[0];
    let bias_4d = ffi::ggml_reshape_4d(ctx, ptr_of(bias), 1, 1, channels, 1);
    ffi::ggml_add(ctx, conv_out, bias_4d)
}

fn ptr_of(slot: TensorSlot) -> *mut ffi::ggml_tensor {
    slot.0
}

fn ensure_tensor(tensor: *mut ffi::ggml_tensor, name: &str) -> Result<()> {
    if tensor.is_null() {
        Err(Error::InvalidTensor(format!(
            "{name} allocation returned null"
        )))
    } else {
        Ok(())
    }
}

fn set_name(tensor: *mut ffi::ggml_tensor, name: &str) -> Result<()> {
    let c_name = CString::new(name)?;
    unsafe { ffi::ggml_set_name(tensor, c_name.as_ptr()) };
    Ok(())
}
