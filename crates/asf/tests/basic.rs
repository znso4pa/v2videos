//! 不依赖真样本的基础测试：坏输入必须**报错而不是 panic**，以及几个纯函数。

use v2v_asf::{fourcc_string, AsfDemuxer, Error};

#[test]
fn rejects_non_asf_input() {
    let data = b"this is definitely not an asf file, it is just some text........";
    assert!(
        matches!(AsfDemuxer::new(&data[..]), Err(Error::Bad(_))),
        "非 ASF 输入应当报 Bad"
    );
}

#[test]
fn truncated_header_is_an_error_not_a_panic() {
    // 只给 Header Object 的 GUID，后面什么都没有。
    let mut data = Vec::new();
    data.extend_from_slice(&[
        0x30, 0x26, 0xB2, 0x75, 0x8E, 0x66, 0xCF, 0x11, 0xA6, 0xD9, 0x00, 0xAA, 0x00, 0x62, 0xCE,
        0x6C,
    ]);
    assert!(AsfDemuxer::new(&data[..]).is_err());
}

#[test]
fn empty_input_is_an_error() {
    assert!(AsfDemuxer::new(&[][..]).is_err());
}

#[test]
fn fourcc_renders_printable_ascii() {
    // "MP43" 小端存为 0x3334504D。
    assert_eq!(fourcc_string(0x3334_504D), "MP43");
    assert_eq!(fourcc_string(0x3356_4D57), "WMV3");
    // 不可打印字节退化成 `.`，绝不产生乱码。
    assert_eq!(fourcc_string(0x0000_0001), "....");
}
