/*
Speaker diarization runner for Sona using NVIDIA Sortformer v2.

Outputs JSON array of speaker segments to stdout.
All diagnostics go to stderr so sona can parse stdout cleanly.

Usage:
    wget https://huggingface.co/altunenes/parakeet-rs/resolve/main/diar_streaming_sortformer_4spk-v2.1.onnx
    wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav
    cargo run diar_streaming_sortformer_4spk-v2.1.onnx 6_speakers.wav

Output (stdout):
  [{"start":0.00,"end":1.50,"speaker_id":0}, ...]
*/

use hound;
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
use serde::Serialize;
use std::env;
use std::process;
use std::time::Instant;

#[derive(Serialize)]
struct Segment {
    start: f32,
    end: f32,
    speaker_id: usize,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: sona-diarize <model.onnx> <audio.wav>");
        process::exit(1);
    }

    let model_path = &args[1];
    let audio_path = &args[2];

    eprintln!("sona-diarize: loading audio {}", audio_path);

    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    eprintln!(
        "sona-diarize: {} samples ({} Hz, {} ch, {:.1}s)",
        audio.len(),
        spec.sample_rate,
        spec.channels,
        audio.len() as f32 / spec.sample_rate as f32 / spec.channels as f32
    );

    eprintln!("sona-diarize: running sortformer on {}", model_path);

    let mut sortformer = Sortformer::with_config(
        model_path,
        None,
        DiarizationConfig::callhome(),
    )?;

    let speaker_segments = sortformer.diarize(audio, spec.sample_rate, spec.channels)?;

    let segments: Vec<Segment> = speaker_segments
        .iter()
        .map(|seg| Segment {
            start: seg.start,
            end: seg.end,
            speaker_id: seg.speaker_id,
        })
        .collect();

    // JSON to stdout — this is what sona reads
    println!("{}", serde_json::to_string(&segments)?);

    eprintln!(
        "sona-diarize: done, {} segments in {:.2}s",
        segments.len(),
        start_time.elapsed().as_secs_f32()
    );

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("sona-diarize: error: {}", e);
        process::exit(1);
    }
}
