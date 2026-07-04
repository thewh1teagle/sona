use std::path::Path;
use std::str::FromStr;

use parakeet_rs::{read_wav_16k_mono, Backend, ParakeetModel};

const MODEL_PATH: &str = "models/parakeet-tdt-0.6b-v3-Q8_0.gguf";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let audio = args
        .next()
        .ok_or("usage: cargo run -p parakeet-rs --example basic -- <audio.wav> [auto|cpu|gpu]")?;
    let backend = args
        .next()
        .map(|value| Backend::from_str(&value))
        .transpose()?
        .unwrap_or_default();

    let samples = read_wav_16k_mono(&audio)?;
    let mut model = ParakeetModel::load_with_backend(Path::new(MODEL_PATH), backend)?;
    let transcript = model.transcribe(&samples)?;
    println!("{}", transcript.text);
    Ok(())
}
