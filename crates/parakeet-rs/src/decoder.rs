use whisper_rs::ffi;

use crate::encoder::EncoderTensor;
use crate::hparams::{HeadKind, ParakeetHParams};
use crate::tokenizer::{normalize_spaces, Tokenizer};
use crate::weights::{LoadedWeights, TensorSlot};
use crate::{Error, Result};

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans_a: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        inc_x: i32,
        beta: f32,
        y: *mut f32,
        inc_y: i32,
    );

    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct RawToken {
    pub(crate) id: i32,
    pub(crate) p: f32,
    pub(crate) step_at_emit: i32,
    pub(crate) duration_frames: i32,
}

struct HostDecoder {
    predictor: HostPredictor,
    joint: HostJoint,
    blank_id: i32,
    tdt_durations: Vec<i32>,
    tdt_max_symbols: i32,
}

struct HostPredictor {
    pred_hidden: usize,
    pred_vocab: usize,
    embed_w: Vec<f32>,
    lstm: Vec<HostLstmLayer>,
}

struct HostLstmLayer {
    wx: Vec<f32>,
    wh: Vec<f32>,
    b: Vec<f32>,
}

struct HostJoint {
    d_enc: usize,
    pred_hidden: usize,
    joint_h: usize,
    joint_n: usize,
    activation: String,
    enc_w: Vec<f32>,
    enc_b: Vec<f32>,
    pred_w: Vec<f32>,
    pred_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

#[derive(Clone)]
struct LstmState {
    h: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,
}

impl LstmState {
    fn new(n_layers: usize, hidden: usize) -> Self {
        Self {
            h: vec![vec![0.0; hidden]; n_layers],
            c: vec![vec![0.0; hidden]; n_layers],
        }
    }
}

pub(crate) fn decode_tdt(
    weights: &LoadedWeights,
    hp: &ParakeetHParams,
    enc: &EncoderTensor,
    tokenizer: &Tokenizer,
) -> Result<(Vec<RawToken>, String)> {
    if hp.head_kind != HeadKind::Tdt {
        return Err(Error::UnsupportedVariant(
            "parakeet-rs currently implements TDT decode only".to_string(),
        ));
    }
    let host = HostDecoder::load(weights, hp)?;
    let raw = host.decode(enc)?;
    let mut ids = Vec::with_capacity(raw.len());
    for token in &raw {
        if !is_strippable_special(tokenizer, token.id) {
            ids.push(token.id);
        }
    }
    let mut text = tokenizer.decode(&ids);
    normalize_spaces(&mut text);
    Ok((raw, text))
}

impl HostDecoder {
    fn load(weights: &LoadedWeights, hp: &ParakeetHParams) -> Result<Self> {
        let t_total = crate::timing::start();
        let predictor_slots = weights
            .slots
            .predictor
            .as_ref()
            .ok_or_else(|| Error::InvalidTensor("missing predictor weights".to_string()))?;
        let joint_slots = weights
            .slots
            .joint
            .as_ref()
            .ok_or_else(|| Error::InvalidTensor("missing joint weights".to_string()))?;

        let pred_hidden = hp.pred_hidden as usize;
        let pred_vocab = hp.pred_vocab as usize;
        let joint_h = hp.joint_hidden as usize;
        let joint_n = (hp.pred_vocab - 1 + hp.joint_num_extra_outputs + 1) as usize;

        let t = crate::timing::start();
        let mut lstm = Vec::with_capacity(hp.pred_n_layers as usize);
        for layer in &predictor_slots.lstm {
            lstm.push(HostLstmLayer {
                wx: read_tensor_to_f32(layer.wx)?,
                wh: read_tensor_to_f32(layer.wh)?,
                b: read_tensor_to_f32(layer.b)?,
            });
        }
        let predictor = HostPredictor {
            pred_hidden,
            pred_vocab,
            embed_w: read_tensor_to_f32(predictor_slots.embed_w)?,
            lstm,
        };
        crate::timing::log("decoder.load.predictor", t);

        let t = crate::timing::start();
        let joint = HostJoint {
            d_enc: hp.enc_d_model as usize,
            pred_hidden,
            joint_h,
            joint_n,
            activation: hp.joint_activation.clone(),
            enc_w: read_tensor_to_f32(joint_slots.enc_w)?,
            enc_b: read_tensor_to_f32(joint_slots.enc_b)?,
            pred_w: read_tensor_to_f32(joint_slots.pred_w)?,
            pred_b: read_tensor_to_f32(joint_slots.pred_b)?,
            out_w: read_tensor_to_f32(joint_slots.out_w)?,
            out_b: read_tensor_to_f32(joint_slots.out_b)?,
        };
        crate::timing::log("decoder.load.joint", t);
        crate::timing::log("decoder.load.total", t_total);

        Ok(Self {
            predictor,
            joint,
            blank_id: hp.pred_vocab - 1,
            tdt_durations: hp.tdt_durations.clone(),
            tdt_max_symbols: hp.tdt_max_symbols,
        })
    }

    fn decode(&self, enc: &EncoderTensor) -> Result<Vec<RawToken>> {
        let t_total = crate::timing::start();
        if enc.d_model != self.joint.d_enc {
            return Err(Error::InvalidTensor(format!(
                "encoder d_model mismatch: got {}, expected {}",
                enc.d_model, self.joint.d_enc
            )));
        }
        let n_layers = self.predictor.lstm.len();
        let hidden = self.predictor.pred_hidden;
        let mut state = LstmState::new(n_layers, hidden);
        let mut next_state = LstmState::new(n_layers, hidden);
        let t = crate::timing::start();
        let enc_proj = self.precompute_enc_proj(enc);
        crate::timing::log("decoder.enc_proj", t);
        let mut out = Vec::new();
        let mut scratch_x = vec![0.0; hidden];
        let mut decoder_out = vec![0.0; hidden];
        let mut logits = vec![0.0; self.joint.joint_n];
        let mut probs = Vec::new();

        let mut last_token = -1;
        let mut step = 0_i32;
        let mut new_symbols = 0_i32;
        let mut predictor_dirty = true;
        let max_iters = 16 * enc.n_frames as i32 + 1024;
        let mut iter = 0_i32;

        let mut predictor_ms = 0.0_f64;
        let mut joint_ms = 0.0_f64;
        let mut confidence_ms = 0.0_f64;
        while step < enc.n_frames as i32 && iter < max_iters {
            iter += 1;
            if predictor_dirty {
                let t = std::time::Instant::now();
                self.predictor_step(last_token, &state, &mut next_state, &mut scratch_x);
                predictor_ms += t.elapsed().as_secs_f64() * 1000.0;
                decoder_out.copy_from_slice(next_state.h.last().expect("predictor layer"));
                predictor_dirty = false;
            }

            let enc_off = step as usize * self.joint.joint_h;
            let t = std::time::Instant::now();
            self.joint_step(
                &enc_proj[enc_off..enc_off + self.joint.joint_h],
                &decoder_out,
                &mut logits,
            );
            joint_ms += t.elapsed().as_secs_f64() * 1000.0;

            let n_token_cls = self.predictor.pred_vocab;
            let pred_token = argmax(&logits[..n_token_cls]) as i32;
            let decision = argmax(&logits[n_token_cls..n_token_cls + self.tdt_durations.len()]);
            let duration = self.tdt_durations[decision];
            let is_blank = pred_token == self.blank_id;

            if !is_blank {
                let t = std::time::Instant::now();
                let p = token_confidence(&logits[..n_token_cls], &mut probs);
                confidence_ms += t.elapsed().as_secs_f64() * 1000.0;
                out.push(RawToken {
                    id: pred_token,
                    p,
                    step_at_emit: step,
                    duration_frames: duration,
                });
                last_token = pred_token;
                std::mem::swap(&mut state, &mut next_state);
                predictor_dirty = true;
            }

            step += duration;
            new_symbols += 1;
            if duration != 0 {
                new_symbols = 0;
            } else if self.tdt_max_symbols > 0 && new_symbols >= self.tdt_max_symbols {
                step += 1;
                new_symbols = 0;
            } else if is_blank && self.tdt_max_symbols > 0 {
                let skip = self.tdt_max_symbols - new_symbols;
                if skip > 0 && iter + skip < max_iters {
                    iter += skip;
                    step += 1;
                    new_symbols = 0;
                }
            }
        }

        if iter >= max_iters {
            return Err(Error::InvalidTensor(format!(
                "TDT decoder hit iteration cap {max_iters}"
            )));
        }
        if crate::timing::enabled() {
            eprintln!(
                "parakeet-rs: decoder.loop iters={iter} tokens={} predictor={predictor_ms:.3} ms joint={joint_ms:.3} ms confidence={confidence_ms:.3} ms",
                out.len()
            );
        }
        crate::timing::log("decoder.total", t_total);
        Ok(out)
    }

    fn predictor_step(
        &self,
        last_token: i32,
        prev: &LstmState,
        next: &mut LstmState,
        scratch_x: &mut [f32],
    ) {
        if last_token < 0 {
            scratch_x.fill(0.0);
        } else {
            let off = last_token as usize * self.predictor.pred_hidden;
            scratch_x
                .copy_from_slice(&self.predictor.embed_w[off..off + self.predictor.pred_hidden]);
        }

        let hidden = self.predictor.pred_hidden;
        let mut input = scratch_x.to_vec();
        let mut gates = vec![0.0; 4 * hidden];
        for (layer_idx, layer) in self.predictor.lstm.iter().enumerate() {
            gates.copy_from_slice(&layer.b);
            matvec_add(&layer.wx, hidden, 4 * hidden, &input, &mut gates);
            matvec_add(
                &layer.wh,
                hidden,
                4 * hidden,
                &prev.h[layer_idx],
                &mut gates,
            );
            for i in 0..hidden {
                let input_gate = sigmoid(gates[i]);
                let forget_gate = sigmoid(gates[hidden + i]);
                let cell_gate = gates[2 * hidden + i].tanh();
                let output_gate = sigmoid(gates[3 * hidden + i]);
                let c = forget_gate * prev.c[layer_idx][i] + input_gate * cell_gate;
                next.c[layer_idx][i] = c;
                next.h[layer_idx][i] = output_gate * c.tanh();
            }
            input.clone_from(&next.h[layer_idx]);
        }
    }

    fn precompute_enc_proj(&self, enc: &EncoderTensor) -> Vec<f32> {
        let mut out = vec![0.0; enc.n_frames * self.joint.joint_h];
        matmat_transb(
            &enc.values,
            &self.joint.enc_w,
            enc.n_frames,
            self.joint.joint_h,
            self.joint.d_enc,
            &mut out,
        );
        for row in out.chunks_exact_mut(self.joint.joint_h) {
            for (value, bias) in row.iter_mut().zip(&self.joint.enc_b) {
                *value += bias;
            }
        }
        out
    }

    fn joint_step(&self, enc_proj: &[f32], pred_state: &[f32], logits: &mut [f32]) {
        let mut joint = self.joint.pred_b.clone();
        matvec_add(
            &self.joint.pred_w,
            self.joint.pred_hidden,
            self.joint.joint_h,
            pred_state,
            &mut joint,
        );
        for (value, enc) in joint.iter_mut().zip(enc_proj) {
            *value += enc;
            *value = match self.joint.activation.as_str() {
                "relu" => value.max(0.0),
                "sigmoid" => sigmoid(*value),
                _ => value.tanh(),
            };
        }
        logits.copy_from_slice(&self.joint.out_b);
        matvec_add(
            &self.joint.out_w,
            self.joint.joint_h,
            self.joint.joint_n,
            &joint,
            logits,
        );
    }
}

fn read_tensor_to_f32(slot: TensorSlot) -> Result<Vec<f32>> {
    let tensor = slot.0;
    if tensor.is_null() {
        return Err(Error::InvalidTensor("null tensor".to_string()));
    }
    let nelem = unsafe { ffi::ggml_nelements(tensor) };
    if nelem <= 0 {
        return Err(Error::InvalidTensor("empty tensor".to_string()));
    }
    let nbytes = unsafe { ffi::ggml_nbytes(tensor) };
    let ty = unsafe { (*tensor).type_ };
    let mut out = vec![0.0_f32; nelem as usize];
    if ty == ffi::ggml_type_GGML_TYPE_F32 {
        unsafe {
            ffi::ggml_backend_tensor_get(tensor, out.as_mut_ptr().cast(), 0, nbytes);
        }
        return Ok(out);
    }

    let traits = unsafe { ffi::ggml_get_type_traits(ty) };
    if traits.is_null() {
        return Err(Error::InvalidTensor("missing ggml type traits".to_string()));
    }
    let Some(to_float) = (unsafe { (*traits).to_float }) else {
        return Err(Error::InvalidTensor(
            "tensor dtype has no to_float conversion".to_string(),
        ));
    };
    let mut raw = vec![0_u8; nbytes];
    unsafe {
        ffi::ggml_backend_tensor_get(tensor, raw.as_mut_ptr().cast(), 0, nbytes);
        to_float(raw.as_ptr().cast(), out.as_mut_ptr(), nelem);
    }
    Ok(out)
}

#[cfg(target_os = "macos")]
fn matvec_add(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.len(), in_dim * out_dim);
    unsafe {
        cblas_sgemv(
            101,
            111,
            out_dim as i32,
            in_dim as i32,
            1.0,
            w.as_ptr(),
            in_dim as i32,
            x.as_ptr(),
            1,
            1.0,
            y.as_mut_ptr(),
            1,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn matvec_add(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.len(), in_dim * out_dim);
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut sum = 0.0_f32;
        for i in 0..in_dim {
            sum += row[i] * x[i];
        }
        y[o] += sum;
    }
}

#[cfg(target_os = "macos")]
fn matmat_transb(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    unsafe {
        cblas_sgemm(
            101,
            111,
            112,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn matmat_transb(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    for row in 0..m {
        let x = &a[row * k..(row + 1) * k];
        let y = &mut c[row * n..(row + 1) * n];
        matvec_add(b, k, n, x, y);
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = values[0];
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > best_v {
            best_i = i;
            best_v = v;
        }
    }
    best_i
}

fn token_confidence(logits: &[f32], probs: &mut Vec<f32>) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    probs.resize(logits.len(), 0.0);
    let mut sum = 0.0_f64;
    for (dst, &logit) in probs.iter_mut().zip(logits) {
        let e = (logit - max).exp();
        *dst = e;
        sum += f64::from(e);
    }
    let mut entropy = 0.0_f64;
    for p in probs {
        let prob = f64::from(*p) / sum;
        entropy -= prob * (prob + 1e-10).ln();
    }
    let max_entropy = (logits.len() as f64).ln();
    if max_entropy <= 0.0 {
        1.0
    } else {
        (1.0 - entropy / max_entropy) as f32
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn is_strippable_special(tokenizer: &Tokenizer, id: i32) -> bool {
    tokenizer.is_control(id)
        || tokenizer.token(id).is_some_and(|piece| {
            let bytes = piece.as_bytes();
            if bytes.len() < 7 || bytes.first() != Some(&b'<') || bytes.last() != Some(&b'>') {
                return false;
            }
            let inner = &piece[1..piece.len() - 1];
            let Some((lang, region)) = inner.split_once('-') else {
                return false;
            };
            (2..=3).contains(&lang.len())
                && lang.bytes().all(|b| b.is_ascii_lowercase())
                && (2..=4).contains(&region.len())
                && region.bytes().all(|b| b.is_ascii_alphabetic())
        })
}
