use rustfft::{num_complex::Complex, FftPlanner};

use crate::hparams::ParakeetHParams;
use crate::{Error, Result};

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
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

const LOG_EPS: f64 = 5.960_464_477_539_063e-8;
const NORM_EPS: f64 = 1.0e-5;
const SLANEY_FSP: f64 = 200.0 / 3.0;
const SLANEY_MIN_LOG_HZ: f64 = 1000.0;

#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    pub values: Vec<f32>,
    pub n_mels: usize,
    pub n_frames: usize,
}

#[derive(Clone)]
pub(crate) struct MelFrontend {
    sample_rate: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    normalize: String,
    pad_mode: PadMode,
    pre_emphasis: f32,
    filterbank: Vec<f32>,
    window: Vec<f64>,
}

impl MelFrontend {
    pub(crate) fn from_hparams(hparams: &ParakeetHParams) -> Result<Self> {
        let sample_rate = usize::try_from(hparams.fe_sample_rate)
            .map_err(|_| Error::InvalidGguf("invalid sample rate".to_string()))?;
        let n_fft = usize::try_from(hparams.fe_n_fft)
            .map_err(|_| Error::InvalidGguf("invalid n_fft".to_string()))?;
        let win_length = usize::try_from(hparams.fe_win_length)
            .map_err(|_| Error::InvalidGguf("invalid win_length".to_string()))?;
        let hop_length = usize::try_from(hparams.fe_hop_length)
            .map_err(|_| Error::InvalidGguf("invalid hop_length".to_string()))?;
        let n_mels = usize::try_from(hparams.fe_num_mels)
            .map_err(|_| Error::InvalidGguf("invalid num_mels".to_string()))?;
        let filterbank = build_mel_filterbank_slaney(
            sample_rate,
            n_fft,
            n_mels,
            f64::from(hparams.fe_f_min),
            f64::from(hparams.fe_f_max),
        );
        let window = build_hann_window_symmetric_padded(win_length, n_fft);

        Ok(Self {
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            normalize: hparams.fe_normalize.clone(),
            pad_mode: PadMode::Constant,
            pre_emphasis: hparams.fe_pre_emphasis,
            filterbank,
            window,
        })
    }

    pub(crate) fn compute(&self, samples: &[f32]) -> Result<MelSpectrogram> {
        if samples.is_empty() {
            return Err(Error::EmptyAudio);
        }

        let n_frames = samples.len() / self.hop_length + 1;
        let n_freq = self.n_fft / 2 + 1;
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        let padded = self.pad_and_preemphasize(samples);
        #[cfg(target_os = "macos")]
        let mut power = vec![0.0_f32; n_frames * n_freq];
        #[cfg(not(target_os = "macos"))]
        let mut spectrum = vec![0.0_f64; n_freq];
        let mut log_mel = vec![0.0_f32; self.n_mels * n_frames];
        let mut frame = vec![Complex::<f64>::new(0.0, 0.0); self.n_fft];

        for t in 0..n_frames {
            let offset = t * self.hop_length;
            for i in 0..self.n_fft {
                frame[i].re = padded[offset + i] * self.window[i];
                frame[i].im = 0.0;
            }
            fft.process(&mut frame);
            #[cfg(target_os = "macos")]
            {
                let row = &mut power[t * n_freq..(t + 1) * n_freq];
                for k in 0..n_freq {
                    let c = frame[k];
                    row[k] = (c.re * c.re + c.im * c.im) as f32;
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                for k in 0..n_freq {
                    let c = frame[k];
                    spectrum[k] = c.re * c.re + c.im * c.im;
                }
                for m in 0..self.n_mels {
                    let filter = &self.filterbank[m * n_freq..(m + 1) * n_freq];
                    let mut sum = 0.0_f64;
                    for k in 0..n_freq {
                        sum += f64::from(filter[k]) * spectrum[k];
                    }
                    log_mel[m * n_frames + t] = (sum + LOG_EPS).ln() as f32;
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    112,
                    self.n_mels as i32,
                    n_frames as i32,
                    n_freq as i32,
                    1.0,
                    self.filterbank.as_ptr(),
                    n_freq as i32,
                    power.as_ptr(),
                    n_freq as i32,
                    0.0,
                    log_mel.as_mut_ptr(),
                    n_frames as i32,
                );
            }
            for v in &mut log_mel {
                *v = (*v as f64 + LOG_EPS).ln() as f32;
            }
        }

        let values = match self.normalize.as_str() {
            "none" => self.mask_trailing_if_needed(log_mel, samples.len(), n_frames),
            "per_feature" => self.normalize_per_feature(&log_mel, n_frames),
            other => {
                return Err(Error::InvalidGguf(format!(
                    "unsupported frontend normalization {other}"
                )))
            }
        };

        Ok(MelSpectrogram {
            values,
            n_mels: self.n_mels,
            n_frames,
        })
    }

    pub(crate) fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn pad_and_preemphasize(&self, samples: &[f32]) -> Vec<f64> {
        let pad = self.n_fft / 2;
        let mut emphasized = vec![0.0_f64; samples.len()];
        if self.pre_emphasis != 0.0 {
            let alpha = f64::from(self.pre_emphasis);
            emphasized[0] = f64::from(samples[0]);
            for i in 1..samples.len() {
                emphasized[i] = f64::from(samples[i]) - alpha * f64::from(samples[i - 1]);
            }
        } else {
            for (dst, src) in emphasized.iter_mut().zip(samples) {
                *dst = f64::from(*src);
            }
        }

        let mut padded = vec![0.0_f64; samples.len() + 2 * pad];
        if self.pad_mode == PadMode::Reflect {
            for i in 0..pad {
                padded[i] = emphasized[pad - i];
            }
        }
        padded[pad..pad + samples.len()].copy_from_slice(&emphasized);
        if self.pad_mode == PadMode::Reflect {
            for i in 0..pad {
                padded[pad + samples.len() + i] = emphasized[samples.len() - 2 - i];
            }
        }
        padded
    }

    fn mask_trailing_if_needed(
        &self,
        mut values: Vec<f32>,
        n_samples: usize,
        n_frames: usize,
    ) -> Vec<f32> {
        let valid = n_samples / self.hop_length;
        for m in 0..self.n_mels {
            let row = &mut values[m * n_frames..(m + 1) * n_frames];
            for value in &mut row[valid..] {
                *value = 0.0;
            }
        }
        values
    }

    fn normalize_per_feature(&self, log_mel: &[f32], n_frames: usize) -> Vec<f32> {
        let mut values = vec![0.0_f32; self.n_mels * n_frames];
        let mask_last = self.pad_mode == PadMode::Constant;
        let n_norm = if mask_last { n_frames - 1 } else { n_frames };
        for m in 0..self.n_mels {
            let src = &log_mel[m * n_frames..(m + 1) * n_frames];
            let dst = &mut values[m * n_frames..(m + 1) * n_frames];
            let norm_src = &src[..n_norm];
            let mean = norm_src.iter().map(|&v| f64::from(v)).sum::<f64>() / n_norm as f64;
            let sumsq = norm_src
                .iter()
                .map(|&v| {
                    let d = f64::from(v) - mean;
                    d * d
                })
                .sum::<f64>();
            let denom = (n_norm.saturating_sub(1)).max(1) as f64;
            let inv = 1.0 / ((sumsq / denom).sqrt() + NORM_EPS);
            for (out, &value) in dst[..n_norm].iter_mut().zip(norm_src) {
                *out = ((f64::from(value) - mean) * inv) as f32;
            }
            if mask_last {
                dst[n_norm] = 0.0;
            }
        }
        values
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PadMode {
    Constant,
    Reflect,
}

fn slaney_hz_to_mel(hz: f64) -> f64 {
    let min_log_mel = SLANEY_MIN_LOG_HZ / SLANEY_FSP;
    let logstep = 6.4_f64.ln() / 27.0;
    if hz < SLANEY_MIN_LOG_HZ {
        hz / SLANEY_FSP
    } else {
        min_log_mel + (hz / SLANEY_MIN_LOG_HZ).ln() / logstep
    }
}

fn slaney_mel_to_hz(mel: f64) -> f64 {
    let min_log_mel = SLANEY_MIN_LOG_HZ / SLANEY_FSP;
    let logstep = 6.4_f64.ln() / 27.0;
    if mel < min_log_mel {
        mel * SLANEY_FSP
    } else {
        SLANEY_MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
    }
}

fn build_mel_filterbank_slaney(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    f_min: f64,
    f_max: f64,
) -> Vec<f32> {
    let n_freq = n_fft / 2 + 1;
    let mut out = vec![0.0_f32; n_mels * n_freq];
    let fft_freqs = (0..n_freq)
        .map(|k| sample_rate as f64 * k as f64 / n_fft as f64)
        .collect::<Vec<_>>();
    let mel_min = slaney_hz_to_mel(f_min);
    let mel_max = slaney_hz_to_mel(f_max);
    let hz_freqs = (0..n_mels + 2)
        .map(|m| {
            let mel = mel_min + (mel_max - mel_min) * m as f64 / (n_mels + 1) as f64;
            slaney_mel_to_hz(mel)
        })
        .collect::<Vec<_>>();

    let fdiff = hz_freqs
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    for m in 0..n_mels {
        let enorm = 2.0 / (hz_freqs[m + 2] - hz_freqs[m]);
        for (k, fft_freq) in fft_freqs.iter().enumerate() {
            let lower = (fft_freq - hz_freqs[m]) / fdiff[m];
            let upper = (hz_freqs[m + 2] - fft_freq) / fdiff[m + 1];
            out[m * n_freq + k] = lower.min(upper).max(0.0).mul_add(enorm, 0.0) as f32;
        }
    }
    out
}

fn build_hann_window_symmetric_padded(win_length: usize, n_fft: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_fft];
    let pad_each = (n_fft - win_length) / 2;
    let denom = (win_length - 1) as f64;
    for k in 0..win_length {
        out[pad_each + k] = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / denom).cos();
    }
    out
}
