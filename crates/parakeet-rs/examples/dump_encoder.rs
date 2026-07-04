use std::fs::File;
use std::io::Write;
use std::path::Path;

use parakeet_rs::{read_wav_16k_mono, Backend, ParakeetModel};

const MODEL_PATH: &str = "models/parakeet-tdt-0.6b-v3-Q8_0.gguf";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let audio = args
        .next()
        .ok_or("usage: cargo run -p parakeet-rs --example dump_encoder -- <audio.wav> <out.f32>")?;
    let out_path = args
        .next()
        .ok_or("usage: cargo run -p parakeet-rs --example dump_encoder -- <audio.wav> <out.f32>")?;

    let samples = read_wav_16k_mono(&audio)?;
    let model = ParakeetModel::load_with_backend(Path::new(MODEL_PATH), Backend::Cpu)?;
    let mel = model.compute_mel(&samples)?;
    let enc = model.compute_encoder(&mel)?;

    let mut out = File::create(out_path)?;
    for value in &enc.values {
        out.write_all(&value.to_le_bytes())?;
    }
    eprintln!("wrote {} x {} encoder", enc.d_model, enc.n_frames);
    Ok(())
}
