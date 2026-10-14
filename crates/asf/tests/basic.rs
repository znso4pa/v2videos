//! 不依赖真样本的基础测试：坏输入必须**报错而不是 panic**，以及几个纯函数。

use v2v_asf::{fourcc_string, AsfDemuxer, Error};

const HEADER_GUID: [u8; 16] = [
    0x30, 0x26, 0xB2, 0x75, 0x8E, 0x66, 0xCF, 0x11, 0xA6, 0xD9, 0x00, 0xAA, 0x00, 0x62, 0xCE, 0x6C,
];
const FILE_PROPERTIES_GUID: [u8; 16] = [
    0xA1, 0xDC, 0xAB, 0x8C, 0x47, 0xA9, 0xCF, 0x11, 0x8E, 0xE4, 0x00, 0xC0, 0x0C, 0x20, 0x53, 0x65,
];

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
    let data = HEADER_GUID.to_vec();
    assert!(AsfDemuxer::new(&data[..]).is_err());
}

/// ⚠️ `Max Packet Size` 是**文件自己声明的**，而 `read_one_packet` 的读法是
/// 「先按它分配整包缓冲」（`vec![0u8; packet_size]`）—— 不挡的话，一个坏文件
/// 声明 4 GiB 就真的会去分配 4 GiB。
///
/// ffmpeg 也没校验这个字段，但它**不按它分配**（包体逐字段流式读），所以没这个问题。
/// 我们的读法是「整包进内存再解析」，这一条得自己挡。
///
/// 这里只需要 File Properties 一个对象 —— 校验在解析它的当下就会触发，
/// 后面有没有 Data Object 都无所谓。
#[test]
fn absurd_max_packet_size_is_rejected_before_allocating() {
    let mut fp = Vec::new();
    fp.extend_from_slice(&[0u8; 16]); // 文件 ID
    fp.extend_from_slice(&0u64.to_le_bytes()); // 文件大小
    fp.extend_from_slice(&0u64.to_le_bytes()); // 创建时间
    fp.extend_from_slice(&1u64.to_le_bytes()); // 包数
    fp.extend_from_slice(&0u64.to_le_bytes()); // 播放时长
    fp.extend_from_slice(&0u64.to_le_bytes()); // 发送时长
    fp.extend_from_slice(&0u32.to_le_bytes()); // preroll
    fp.extend_from_slice(&0u32.to_le_bytes()); // ignore
    fp.extend_from_slice(&2u32.to_le_bytes()); // flags（非广播）
    fp.extend_from_slice(&3200u32.to_le_bytes()); // min packet size（合法）
    fp.extend_from_slice(&u32::MAX.to_le_bytes()); // max packet size ← 坏字段
    fp.extend_from_slice(&0u32.to_le_bytes()); // max bitrate

    let mut data = Vec::new();
    data.extend_from_slice(&HEADER_GUID);
    data.extend_from_slice(&(30u64 + 24 + fp.len() as u64).to_le_bytes()); // 头总长
    data.extend_from_slice(&1u32.to_le_bytes()); // 子对象数
    data.push(1);
    data.push(2);
    data.extend_from_slice(&FILE_PROPERTIES_GUID);
    data.extend_from_slice(&((fp.len() + 24) as u64).to_le_bytes());
    data.extend_from_slice(&fp);

    assert!(
        matches!(AsfDemuxer::new(&data[..]), Err(Error::Bad(_))),
        "天文数字的 max_packet_size 必须报 Bad —— 不能照着它去分配"
    );
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
