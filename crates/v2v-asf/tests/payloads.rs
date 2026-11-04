//! ASF **数据包载荷**的回归测试 —— 重点是「一个包里混装两种载荷」这条路径。
//!
//! ## 为什么要有这条
//!
//! 真样本（一个 WMA v2 音频 + MP43 视频的 `.asf`）**第一个包就解不动**：
//! 音频走的是**压缩载荷**（`replicated data length == 1`），视频是**常规载荷**
//! （`replicated == 8`），两者在**同一个包**里。压缩载荷的子载荷读完后，
//! `time_start` 没归零 + 没有把「试探读掉的那一个长度字节」退回去，于是视频载荷的
//! **流号字节被当成了子载荷长度** —— 位流从此错位，直接读到包尾之外。
//!
//! 我们原来的样本只有一个纯视频文件，走不到这条路，所以一直没暴露。
//! 这条测试**手写**一个这样的包（不用任何真样本，夹具不含样本派生数据），
//! 把行为钉住：包尾必须正确切回「读载荷头」，两个载荷都要原样吐出来。
//!
//! 结构（对照 `ffmpeg` 的 `asfdec_f.c`：`asf_get_packet` / `asf_read_frame_header`
//! / `asf_parse_packet` 的 1192-1206 行压缩载荷分支）：

use std::io::Cursor;

use v2v_asf::{AsfDemuxer, StreamKind};

const PACKET_SIZE: u32 = 64;

fn u16le(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn u32le(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn u64le(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}

/// 一个 ASF 对象：GUID + 总长 + 主体。
fn object(guid: &[u8; 16], body: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(guid);
    u64le(&mut v, (body.len() + 24) as u64);
    v.extend_from_slice(body);
    v
}

const HEADER_GUID: [u8; 16] = [
    0x30, 0x26, 0xB2, 0x75, 0x8E, 0x66, 0xCF, 0x11, 0xA6, 0xD9, 0x00, 0xAA, 0x00, 0x62, 0xCE, 0x6C,
];
const DATA_GUID: [u8; 16] = [
    0x36, 0x26, 0xB2, 0x75, 0x8E, 0x66, 0xCF, 0x11, 0xA6, 0xD9, 0x00, 0xAA, 0x00, 0x62, 0xCE, 0x6C,
];
const FILE_PROPERTIES_GUID: [u8; 16] = [
    0xA1, 0xDC, 0xAB, 0x8C, 0x47, 0xA9, 0xCF, 0x11, 0x8E, 0xE4, 0x00, 0xC0, 0x0C, 0x20, 0x53, 0x65,
];
const STREAM_PROPERTIES_GUID: [u8; 16] = [
    0x91, 0x07, 0xDC, 0xB7, 0xB7, 0xA9, 0xCF, 0x11, 0x8E, 0xE6, 0x00, 0xC0, 0x0C, 0x20, 0x53, 0x65,
];
const AUDIO_MEDIA_GUID: [u8; 16] = [
    0x40, 0x9E, 0x69, 0xF8, 0x4D, 0x5B, 0xCF, 0x11, 0xA8, 0xFD, 0x00, 0x80, 0x5F, 0x5C, 0x44, 0x2B,
];
const AUDIO_CONCEAL_SPREAD_GUID: [u8; 16] = [
    0x50, 0xCD, 0xC3, 0xBF, 0x8F, 0x61, 0xCF, 0x11, 0x8B, 0xB2, 0x00, 0xAA, 0x00, 0xB4, 0xE2, 0x20,
];
const VIDEO_MEDIA_GUID: [u8; 16] = [
    0xC0, 0xEF, 0x19, 0xBC, 0x4D, 0x5B, 0xCF, 0x11, 0xA8, 0xFD, 0x00, 0x80, 0x5F, 0x5C, 0x44, 0x2B,
];
const VIDEO_CONCEAL_NONE_GUID: [u8; 16] = [
    0x00, 0x57, 0xFB, 0x20, 0x55, 0x5B, 0xCF, 0x11, 0xA8, 0xFD, 0x00, 0x80, 0x5F, 0x5C, 0x44, 0x2B,
];

/// 造一个「音频（压缩载荷）+ 视频（常规载荷）同包」的最小 ASF。
fn mixed_payload_asf() -> Vec<u8> {
    // ---- 音频流：WMA v2，8 kHz 单声道（字段值只是「像真的」，解码用不到）----
    let mut audio_type = Vec::new();
    u16le(&mut audio_type, 0x0161); // WMA v2
    u16le(&mut audio_type, 1); // 单声道
    u32le(&mut audio_type, 8000);
    u32le(&mut audio_type, 16); // avg bytes/s
    u16le(&mut audio_type, 1); // block align
    u16le(&mut audio_type, 16); // bits
    u16le(&mut audio_type, 0); // cbSize
    let mut sp_audio = Vec::new();
    sp_audio.extend_from_slice(&AUDIO_MEDIA_GUID);
    sp_audio.extend_from_slice(&AUDIO_CONCEAL_SPREAD_GUID);
    u64le(&mut sp_audio, 0); // 时间偏移
    u32le(&mut sp_audio, audio_type.len() as u32);
    u32le(&mut sp_audio, 0); // 纠错数据长度
    u16le(&mut sp_audio, 1); // 流号
    u32le(&mut sp_audio, 0);
    sp_audio.extend_from_slice(&audio_type);

    // ---- 视频流：MP43 16×16 ----
    let mut video_type = Vec::new();
    u32le(&mut video_type, 16);
    u32le(&mut video_type, 16);
    video_type.push(2);
    u16le(&mut video_type, 40);
    u32le(&mut video_type, 40); // biSize
    u32le(&mut video_type, 16);
    u32le(&mut video_type, 16);
    u16le(&mut video_type, 1);
    u16le(&mut video_type, 24);
    video_type.extend_from_slice(b"MP43");
    u32le(&mut video_type, 16 * 16 * 3);
    for _ in 0..4 {
        u32le(&mut video_type, 0);
    }
    let mut sp_video = Vec::new();
    sp_video.extend_from_slice(&VIDEO_MEDIA_GUID);
    sp_video.extend_from_slice(&VIDEO_CONCEAL_NONE_GUID);
    u64le(&mut sp_video, 0);
    u32le(&mut sp_video, video_type.len() as u32);
    u32le(&mut sp_video, 0);
    u16le(&mut sp_video, 2); // 流号
    u32le(&mut sp_video, 0);
    sp_video.extend_from_slice(&video_type);

    // ---- 数据包 ----
    let video_data = [0x11u8, 0x22, 0x33, 0x44];
    let mut pkt = Vec::new();
    pkt.extend_from_slice(&[0x82, 0x00, 0x00]); // EC：标准 3 字节
    pkt.push(0x09); // 长度类型：多载荷 + padding 长度 1 字节
    pkt.push(0x5D); // 属性：对象号 1B / 偏移 4B / replicated 1B / 流号 1B
    pkt.push(0x00); // padding 长度字段（占位，最后回填）
    let pad_len_at = pkt.len() - 1;
    u32le(&mut pkt, 0); // 发送时间
    u16le(&mut pkt, 0); // 时长
    pkt.push(0x82); // 载荷数 2（0x80 | 2）

    // 载荷 1：音频，压缩（replicated = 1）
    pkt.push(0x01); // 流 1（音频），非关键帧
    pkt.push(0x01); // 对象号
    u32le(&mut pkt, 1000); // 这个字段在压缩载荷里当「起始时间戳」
    pkt.push(0x01); // replicated 长度 = 1
    pkt.push(0x00); // 时间增量
    let sub = [0x04u8, 0xAA, 0xBB, 0xCC, 0xDD]; // 子载荷：长度字节 + 数据
    u16le(&mut pkt, sub.len() as u16); // 载荷长
    pkt.extend_from_slice(&sub);

    // 载荷 2：视频，常规（replicated = 8）—— 紧跟在压缩载荷之后，**没有**子载荷长度字节
    pkt.push(0x82); // 流 2 + 关键帧位
    pkt.push(0x01); // 对象号
    u32le(&mut pkt, 0); // 对象内偏移
    pkt.push(0x08); // replicated 长度 = 8
    u32le(&mut pkt, video_data.len() as u32); // 对象总长
    u32le(&mut pkt, 1000); // 显示时间
    u16le(&mut pkt, video_data.len() as u16); // 载荷长
    pkt.extend_from_slice(&video_data);

    let pad = PACKET_SIZE as usize - pkt.len();
    assert!(pad >= 2, "包太小，装不下两个载荷（padding {pad}）");
    pkt[pad_len_at] = (pad - 1) as u8;
    pkt.resize(PACKET_SIZE as usize, 0);

    // ---- 头 ----
    // 先把三个子对象拼出来，头部长度按**实际长度**算（别预测）。
    let mut fp = Vec::new();
    fp.extend_from_slice(&[0u8; 16]); // 文件 ID
    u64le(&mut fp, 0); // 文件大小，最后回填
    u64le(&mut fp, 0); // 创建时间
    u64le(&mut fp, 1); // 包数
    u64le(&mut fp, 0); // 发送时长
    u64le(&mut fp, 0); // 播放时长
    u64le(&mut fp, 0); // preroll = 0（时间戳不做偏移，好断言）
    u32le(&mut fp, 2); // 非广播
    u32le(&mut fp, PACKET_SIZE);
    u32le(&mut fp, PACKET_SIZE);
    u32le(&mut fp, 0xFFFF_FFFF);

    // 头长按各部分的**长度**算（与内容无关），算出来再回填文件大小。
    let header_size =
        30 + (24 + fp.len()) + (24 + sp_audio.len()) + (24 + sp_video.len());
    let file_size = header_size as u64 + 50 + PACKET_SIZE as u64;
    fp[16..24].copy_from_slice(&file_size.to_le_bytes());

    let mut header_body = Vec::new();
    header_body.extend_from_slice(&object(&FILE_PROPERTIES_GUID, &fp));
    header_body.extend_from_slice(&object(&STREAM_PROPERTIES_GUID, &sp_audio));
    header_body.extend_from_slice(&object(&STREAM_PROPERTIES_GUID, &sp_video));
    let mut out = Vec::new();
    out.extend_from_slice(&HEADER_GUID);
    u64le(&mut out, header_size as u64);
    u32le(&mut out, 3); // 子对象数
    out.push(1);
    out.push(2);
    out.extend_from_slice(&header_body);

    out.extend_from_slice(&DATA_GUID);
    u64le(&mut out, 50 + PACKET_SIZE as u64);
    out.extend_from_slice(&[0u8; 16]); // 文件 ID
    u64le(&mut out, 1); // 包数
    out.push(1);
    out.push(1);
    out.extend_from_slice(&pkt);
    assert_eq!(out.len() as u64, file_size);
    out
}

/// 混装载荷：压缩（音频）+ 常规（视频）**在同一个包里**，两个样本都要原样出来。
#[test]
fn compressed_and_normal_payloads_in_one_packet() {
    let file = mixed_payload_asf();
    let mut d = AsfDemuxer::new(Cursor::new(&file)).expect("解析头");
    assert_eq!(d.streams().len(), 2);
    assert_eq!(d.streams()[0].kind, StreamKind::Audio);
    assert_eq!(d.streams()[1].kind, StreamKind::Video);

    let mut got = Vec::new();
    while let Some(s) = d.next_sample().expect("读样本") {
        got.push(s);
    }
    assert_eq!(got.len(), 2, "一个包里的两个载荷都该吐出来：{got:?}");

    // 音频：压缩载荷，4 字节数据；时间戳来自「起始时间戳」字段
    assert_eq!(got[0].stream, 1);
    assert_eq!(got[0].data, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    assert_eq!(got[0].timestamp_ms, 1000);

    // 视频：常规载荷 —— **关键帧位**要保留（错位的实现会把它吃掉）
    assert_eq!(got[1].stream, 2);
    assert!(got[1].key_frame, "视频载荷的关键帧位丢了");
    assert_eq!(got[1].data, vec![0x11, 0x22, 0x33, 0x44]);
    assert_eq!(got[1].timestamp_ms, 1000);
}
