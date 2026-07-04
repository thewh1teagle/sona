use crate::gguf::Gguf;
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub(crate) struct Tokenizer {
    tokens: Vec<String>,
    token_type: Vec<i32>,
}

impl Tokenizer {
    pub(crate) fn load(gguf: &Gguf) -> Result<Self> {
        let model = gguf.required_str("tokenizer.ggml.model")?;
        if model != "unigram" && model != "bpe" {
            return Err(Error::InvalidGguf(format!(
                "unsupported tokenizer model {model}; expected SentencePiece unigram/bpe"
            )));
        }
        let tokens = gguf.required_str_array("tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err(Error::InvalidGguf("tokenizer has no tokens".to_string()));
        }
        let token_type = gguf
            .optional_int_array("tokenizer.ggml.token_type")?
            .unwrap_or_default();
        if !token_type.is_empty() && token_type.len() != tokens.len() {
            return Err(Error::InvalidGguf(
                "tokenizer.ggml.token_type length does not match tokens".to_string(),
            ));
        }
        Ok(Self { tokens, token_type })
    }

    pub(crate) fn decode(&self, ids: &[i32]) -> String {
        let mut out = Vec::<u8>::with_capacity(ids.len() * 4);
        for &id in ids {
            let Some(piece) = self.token(id) else {
                continue;
            };
            if let Some(byte) = byte_fallback(piece) {
                out.push(byte);
                continue;
            }
            let bytes = piece.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i..].starts_with("▁".as_bytes()) {
                    out.push(b' ');
                    i += "▁".len();
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    pub(crate) fn token(&self, id: i32) -> Option<&str> {
        (id >= 0)
            .then_some(id as usize)
            .and_then(|idx| self.tokens.get(idx))
            .map(String::as_str)
    }

    pub(crate) fn is_control(&self, id: i32) -> bool {
        if id < 0 {
            return false;
        }
        self.token_type.get(id as usize).is_some_and(|&ty| ty == 3)
    }
}

fn byte_fallback(piece: &str) -> Option<u8> {
    let b = piece.as_bytes();
    if b.len() != 6 || b[0] != b'<' || b[1] != b'0' || b[2] != b'x' || b[5] != b'>' {
        return None;
    }
    let hi = hex(b[3])?;
    let lo = hex(b[4])?;
    Some((hi << 4) | lo)
}

fn hex(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

pub(crate) fn normalize_spaces(text: &mut String) {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = false;
    for ch in text.chars() {
        let is_space = ch == ' ';
        if is_space && prev_space {
            continue;
        }
        out.push(ch);
        prev_space = is_space;
    }
    *text = out.trim().to_string();
}
