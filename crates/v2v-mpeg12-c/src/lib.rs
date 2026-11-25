//! MPEG-1/2 的可选 C/NEON 内核。宿主仍通过原来的播放器和 C ABI 访问解码器。

#[cfg(v2v_c_mc8)]
mod mc8;

/// C 内核是否已编入且可以在当前 CPU 上运行。
pub fn available() -> bool {
    #[cfg(v2v_c_mc8)]
    return mc8::available();
    #[cfg(not(v2v_c_mc8))]
    false
}

/// 在支持的平台上为进程注册 C 预测内核。
pub fn install() {
    #[cfg(v2v_c_mc8)]
    if available() {
        v2v_mpeg12::decode::set_mc8("c-neon", mc8::predict);
    }
}
