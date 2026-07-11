use nemotron_rs::{MelConfig, MelFrontend, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: transcribe <model.gguf>")?;
    let audio = std::env::args().nth(2);
    let repeat = std::env::args().nth(3).and_then(|value| value.parse().ok()).unwrap_or(1);
    let model = Model::load(path)?;
    let samples = if let Some(audio) = audio {
        let mut reader = hound::WavReader::open(audio)?;
        if reader.spec().sample_rate != 16_000 || reader.spec().channels != 1 {
            return Err("audio must be 16 kHz mono".into());
        }
        reader
            .samples::<i16>()
            .map(|x| x.map(|v| v as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![0.0; 16_000]
    };
    let mut mel_ms = 0.0;
    let mut encode_ms = 0.0;
    let mut decode_ms = 0.0;
    let mut result = None;
    for _ in 0..repeat {
        let t = std::time::Instant::now();
        let mel = MelFrontend::new(MelConfig::default())?.compute(&samples)?;
        mel_ms += t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let (encoded, shape) = model.encode(&mel, model.prompt_id("en-US")?)?;
        encode_ms += t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let tokens = model.decode(&encoded, shape[1] as usize)?;
        decode_ms += t.elapsed().as_secs_f64() * 1000.0;
        let ids = tokens.iter().map(|token| token.id).collect::<Vec<_>>();
        result = Some((model.tokenizer().decode_clean(&ids), tokens));
    }
    eprintln!(
        "average: mel={:.2} encode={:.2} decode={:.2} total={:.2} ms",
        mel_ms / repeat as f64,
        encode_ms / repeat as f64,
        decode_ms / repeat as f64,
        (mel_ms + encode_ms + decode_ms) / repeat as f64
    );
    let (text, tokens) = result.unwrap();
    let ids = tokens.iter().map(|token| token.id).collect::<Vec<_>>();
    println!("tokens={ids:?}\ntext={text:?}");
    Ok(())
}
