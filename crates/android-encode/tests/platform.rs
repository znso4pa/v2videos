//! 非 Android 平台上的行为钉子：API 在、调用明确报 `Unsupported`。
//!
//! 真机（Android）上的行为测试在 `hw.rs`，只在设备/模拟器上编译。

use v2v_android::Mp4HwWriter;

#[test]
fn non_android_reports_unsupported() {
    let out = std::env::temp_dir().join(format!("v2v_hw_stub_{}.mp4", std::process::id()));
    let Err(err) = Mp4HwWriter::open(&out, 1280, 720, 4_000_000, 30, 4, false) else {
        panic!("桩实现不该成功");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("MediaCodec") || msg.contains("Android"),
        "报错要说清是平台不支持，而不是别的：{msg}"
    );
    assert!(!out.exists(), "桩实现不该创建文件");
}
