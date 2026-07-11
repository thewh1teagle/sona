use nemotron_rs::{MelConfig, MelFrontend, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: encode <model.gguf>")?;
    let model = Model::load(path)?;
    let mel = MelFrontend::new(MelConfig::default())?.compute(&vec![0.0; 16_000])?;
    let (values, shape) = model.encode(&mel, 0)?;
    println!(
        "shape={shape:?} finite={} min={} max={}",
        values.iter().all(|x| x.is_finite()),
        values.iter().copied().fold(f32::INFINITY, f32::min),
        values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
    Ok(())
}
