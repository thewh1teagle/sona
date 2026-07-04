use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use parakeet_rs::{read_wav_16k_mono, Backend, ParakeetModel};

const MODEL_PATH: &str = "models/parakeet-tdt-0.6b-v3-Q8_0.gguf";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let audio = args.next().ok_or(
        "usage: cargo run -p parakeet-rs --example bench --release -- <audio.wav> [auto|cpu|gpu] [runs]",
    )?;
    let backend = args
        .next()
        .map(|value| Backend::from_str(&value))
        .transpose()?
        .unwrap_or_default();
    let runs = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(3);
    let warmup = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(0);

    let samples = read_wav_16k_mono(&audio)?;
    let mut model = ParakeetModel::load_with_backend(Path::new(MODEL_PATH), backend)?;
    let mut last = String::new();
    for _ in 0..warmup {
        last = model.transcribe(&samples)?.text;
    }
    let start = Instant::now();
    for _ in 0..runs {
        last = model.transcribe(&samples)?.text;
    }
    let elapsed = start.elapsed();
    println!(
        "runs={runs} total_ms={:.3} avg_ms={:.3}",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / runs as f64
    );
    println!("{last}");
    Ok(())
}
