//! 可选的原生音频解码后端，Rust API 与纯 Rust 后端使用相同的 PCM 刻度。
#![deny(unsafe_code)]

#[cfg(feature = "mpg123")]
#[allow(unsafe_code)]
mod mpg123;

#[cfg(feature = "mpg123")]
pub use mpg123::Mp3Decoder;
