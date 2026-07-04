use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;

use whisper_rs::ffi;

use crate::{Error, Result};

pub(crate) struct Gguf {
    ctx: *mut ffi::gguf_context,
    path: PathBuf,
}

impl Gguf {
    pub(crate) fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let c_path = CString::new(path.to_string_lossy().as_bytes())?;
        let params = ffi::gguf_init_params {
            no_alloc: true,
            ctx: ptr::null_mut(),
        };
        let ctx = unsafe { ffi::gguf_init_from_file(c_path.as_ptr(), params) };
        if ctx.is_null() {
            return Err(Error::LoadGguf(path.to_path_buf()));
        }
        Ok(Self {
            ctx,
            path: path.to_path_buf(),
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn required_str(&self, key: &str) -> Result<String> {
        let key_id = self
            .find_key(key)?
            .ok_or_else(|| Error::MissingKey(key.to_string()))?;
        let ty = unsafe { ffi::gguf_get_kv_type(self.ctx, key_id) };
        if ty != ffi::gguf_type_GGUF_TYPE_STRING {
            return Err(Error::WrongKeyType {
                key: key.to_string(),
                actual: gguf_type_name(ty),
            });
        }
        let ptr = unsafe { ffi::gguf_get_val_str(self.ctx, key_id) };
        Ok(cstr(ptr).to_string())
    }

    pub(crate) fn optional_str(&self, key: &str) -> Result<Option<String>> {
        let Some(key_id) = self.find_key(key)? else {
            return Ok(None);
        };
        let ty = unsafe { ffi::gguf_get_kv_type(self.ctx, key_id) };
        if ty != ffi::gguf_type_GGUF_TYPE_STRING {
            return Err(Error::WrongKeyType {
                key: key.to_string(),
                actual: gguf_type_name(ty),
            });
        }
        let ptr = unsafe { ffi::gguf_get_val_str(self.ctx, key_id) };
        Ok(Some(cstr(ptr).to_string()))
    }

    pub(crate) fn has_key(&self, key: &str) -> Result<bool> {
        Ok(self.find_key(key)?.is_some())
    }

    pub(crate) fn required_i32(&self, key: &str) -> Result<i32> {
        let key_id = self
            .find_key(key)?
            .ok_or_else(|| Error::MissingKey(key.to_string()))?;
        self.i32_at(key, key_id)
    }

    pub(crate) fn optional_i32(&self, key: &str, default: i32) -> Result<i32> {
        let Some(key_id) = self.find_key(key)? else {
            return Ok(default);
        };
        self.i32_at(key, key_id)
    }

    pub(crate) fn required_f32(&self, key: &str) -> Result<f32> {
        let key_id = self
            .find_key(key)?
            .ok_or_else(|| Error::MissingKey(key.to_string()))?;
        let ty = unsafe { ffi::gguf_get_kv_type(self.ctx, key_id) };
        if ty != ffi::gguf_type_GGUF_TYPE_FLOAT32 {
            return Err(Error::WrongKeyType {
                key: key.to_string(),
                actual: gguf_type_name(ty),
            });
        }
        Ok(unsafe { ffi::gguf_get_val_f32(self.ctx, key_id) })
    }

    pub(crate) fn optional_bool(&self, key: &str, default: bool) -> Result<bool> {
        let Some(key_id) = self.find_key(key)? else {
            return Ok(default);
        };
        let ty = unsafe { ffi::gguf_get_kv_type(self.ctx, key_id) };
        if ty != ffi::gguf_type_GGUF_TYPE_BOOL {
            return Err(Error::WrongKeyType {
                key: key.to_string(),
                actual: gguf_type_name(ty),
            });
        }
        Ok(unsafe { ffi::gguf_get_val_bool(self.ctx, key_id) })
    }

    pub(crate) fn optional_int_array(&self, key: &str) -> Result<Option<Vec<i32>>> {
        let Some(key_id) = self.find_key(key)? else {
            return Ok(None);
        };
        let ty = unsafe { ffi::gguf_get_kv_type(self.ctx, key_id) };
        if ty != ffi::gguf_type_GGUF_TYPE_ARRAY {
            return Err(Error::WrongKeyType {
                key: key.to_string(),
                actual: gguf_type_name(ty),
            });
        }
        let arr_ty = unsafe { ffi::gguf_get_arr_type(self.ctx, key_id) };
        if arr_ty != ffi::gguf_type_GGUF_TYPE_INT32 {
            return Err(Error::WrongKeyType {
                key: key.to_string(),
                actual: format!("array<{}>", gguf_type_name(arr_ty)),
            });
        }
        let n = unsafe { ffi::gguf_get_arr_n(self.ctx, key_id) };
        let ptr = unsafe { ffi::gguf_get_arr_data(self.ctx, key_id) }.cast::<i32>();
        if ptr.is_null() && n > 0 {
            return Err(Error::InvalidGguf(format!("{key} array data is null")));
        }
        let values = unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec();
        Ok(Some(values))
    }

    pub(crate) fn optional_str_array(&self, key: &str) -> Result<Vec<String>> {
        let Some(key_id) = self.find_key(key)? else {
            return Ok(Vec::new());
        };
        self.str_array_at(key, key_id)
    }

    pub(crate) fn required_str_array(&self, key: &str) -> Result<Vec<String>> {
        let key_id = self
            .find_key(key)?
            .ok_or_else(|| Error::MissingKey(key.to_string()))?;
        self.str_array_at(key, key_id)
    }

    fn str_array_at(&self, key: &str, key_id: i64) -> Result<Vec<String>> {
        let ty = unsafe { ffi::gguf_get_kv_type(self.ctx, key_id) };
        if ty != ffi::gguf_type_GGUF_TYPE_ARRAY {
            return Err(Error::WrongKeyType {
                key: key.to_string(),
                actual: gguf_type_name(ty),
            });
        }
        let arr_ty = unsafe { ffi::gguf_get_arr_type(self.ctx, key_id) };
        if arr_ty != ffi::gguf_type_GGUF_TYPE_STRING {
            return Err(Error::WrongKeyType {
                key: key.to_string(),
                actual: format!("array<{}>", gguf_type_name(arr_ty)),
            });
        }
        let n = unsafe { ffi::gguf_get_arr_n(self.ctx, key_id) };
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            let ptr = unsafe { ffi::gguf_get_arr_str(self.ctx, key_id, i) };
            values.push(cstr(ptr).to_string());
        }
        Ok(values)
    }

    pub(crate) fn tensors(&self) -> Vec<TensorInfo> {
        let n = unsafe { ffi::gguf_get_n_tensors(self.ctx) };
        let mut tensors = Vec::with_capacity(n.max(0) as usize);
        for id in 0..n {
            let name = cstr(unsafe { ffi::gguf_get_tensor_name(self.ctx, id) }).to_string();
            let ty = unsafe { ffi::gguf_get_tensor_type(self.ctx, id) };
            let offset = unsafe { ffi::gguf_get_tensor_offset(self.ctx, id) };
            let size = unsafe { ffi::gguf_get_tensor_size(self.ctx, id) };
            tensors.push(TensorInfo {
                name,
                ggml_type: ty,
                offset,
                size,
            });
        }
        tensors
    }

    fn find_key(&self, key: &str) -> Result<Option<i64>> {
        let c_key = CString::new(key)?;
        let id = unsafe { ffi::gguf_find_key(self.ctx, c_key.as_ptr()) };
        Ok((id >= 0).then_some(id))
    }

    fn i32_at(&self, key: &str, key_id: i64) -> Result<i32> {
        let ty = unsafe { ffi::gguf_get_kv_type(self.ctx, key_id) };
        match ty {
            ffi::gguf_type_GGUF_TYPE_UINT32 => {
                let value = unsafe { ffi::gguf_get_val_u32(self.ctx, key_id) };
                i32::try_from(value).map_err(|_| Error::InvalidGguf(format!("{key} exceeds i32")))
            }
            ffi::gguf_type_GGUF_TYPE_INT32 => {
                Ok(unsafe { ffi::gguf_get_val_i32(self.ctx, key_id) })
            }
            _ => Err(Error::WrongKeyType {
                key: key.to_string(),
                actual: gguf_type_name(ty),
            }),
        }
    }
}

impl Drop for Gguf {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { ffi::gguf_free(self.ctx) };
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct TensorInfo {
    pub(crate) name: String,
    pub(crate) ggml_type: ffi::ggml_type,
    pub(crate) offset: usize,
    pub(crate) size: usize,
}

fn cstr<'a>(ptr: *const std::os::raw::c_char) -> &'a str {
    if ptr.is_null() {
        return "";
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or("")
}

fn gguf_type_name(ty: ffi::gguf_type) -> String {
    cstr(unsafe { ffi::gguf_type_name(ty) }).to_string()
}
