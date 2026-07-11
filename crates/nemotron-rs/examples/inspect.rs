use nemotron_rs::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: inspect <model.gguf>")?;
    let model = Model::load(path)?;
    println!("{:#?}", model.info());
    Ok(())
}
