use std::time::Instant;

pub(crate) fn enabled() -> bool {
    std::env::var_os("PARAKEET_RS_TIMINGS").is_some()
}

pub(crate) fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

pub(crate) fn log(label: &str, start: Option<Instant>) {
    if let Some(start) = start {
        eprintln!(
            "parakeet-rs: {label}={:.3} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
