use std::ffi::CStr;
use std::str::FromStr;

use whisper_rs::ffi;

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

impl Backend {
    pub(crate) fn init(self) -> Result<ffi::ggml_backend_t> {
        let backend = unsafe {
            match self {
                Self::Auto => ffi::ggml_backend_init_best(),
                Self::Cpu => ffi::ggml_backend_init_by_type(
                    ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU,
                    std::ptr::null(),
                ),
                Self::Gpu => ffi::ggml_backend_init_by_type(
                    ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU,
                    std::ptr::null(),
                )
                .or_else_null(|| {
                    ffi::ggml_backend_init_by_type(
                        ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU,
                        std::ptr::null(),
                    )
                })
                .or_else_null(|| {
                    ffi::ggml_backend_init_by_type(
                        ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_ACCEL,
                        std::ptr::null(),
                    )
                }),
            }
        };
        if backend.is_null() {
            return Err(Error::BackendUnavailable(format!("{self:?}")));
        }
        if self == Self::Cpu {
            unsafe {
                ffi::ggml_backend_cpu_set_n_threads(backend, cpu_threads());
            }
        }
        Ok(backend)
    }
}

trait BackendPtrExt {
    fn or_else_null<F>(self, fallback: F) -> Self
    where
        F: FnOnce() -> Self;
}

impl BackendPtrExt for ffi::ggml_backend_t {
    fn or_else_null<F>(self, fallback: F) -> Self
    where
        F: FnOnce() -> Self,
    {
        if self.is_null() {
            fallback()
        } else {
            self
        }
    }
}

fn cpu_threads() -> i32 {
    if let Ok(value) = std::env::var("PARAKEET_RS_THREADS") {
        if let Ok(parsed) = value.parse::<i32>() {
            if parsed > 0 {
                return parsed;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get().min(8) as i32)
        .unwrap_or(1)
}

impl FromStr for Backend {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" | "best" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            other => Err(Error::BackendUnavailable(format!(
                "unknown backend {other}; expected auto, cpu, or gpu"
            ))),
        }
    }
}

pub(crate) fn backend_name(backend: ffi::ggml_backend_t) -> String {
    if backend.is_null() {
        return "unknown".to_string();
    }
    let ptr = unsafe { ffi::ggml_backend_name(backend) };
    if ptr.is_null() {
        "unknown".to_string()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}
