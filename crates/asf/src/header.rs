//! Header Object 及其子对象的解析。
//!
//! 对象统一是 `GUID(16) + Size(u64) + Payload(Size-24)`。`Size` 含这 24 字节头。
//! 主循环一直读到 **Data Object** 为止 —— 这是 ASF 的布局：Header Object 之后紧跟
//! Data Object，再往后才是可选的索引对象。

use std::io::Read;

use crate::cursor::{Cursor, StreamReader};
use crate::error::{Error, Result};
use crate::guid;
use crate::types::{AudioFormat, FileInfo, PayloadExt, Stream, StreamKind, VideoFormat};

/// 单个对象 payload 的上限。防一个坏 `Size` 直接把内存吃光。
const MAX_OBJECT: usize = 16 * 1024 * 1024;

/// `Max Packet Size` 的上限。
///
/// 这个字段是**文件自己声明的**，而 `read_one_packet` 的读法是
/// 「先按它分配整包缓冲」（`vec![0u8; packet_size]`）—— 不挡的话，一个坏文件
/// 声明 4 GiB 就真的会去分配 4 GiB。
///
/// ⚠️ **ffmpeg 也有这个字段不校验**（它只查 `min_pktsize`），但它**不按它分配**：
/// 包体是逐字段从流里读的，`max_pktsize` 只当默认值与 seek 数学用。我们的读法是
/// 「整包进内存再解析」，所以这一条得自己挡。
///
/// 取值：与 `MAX_OBJECT` 同一个量级（同一个理由）。参照 ffmpeg 的 ASF **封装器**
/// 自己的上限是 65536（`asfenc.c` 的 `PACKET_SIZE_MAX`）、默认 3200 —— 16 MiB 是
/// 它们的 256× / 5000×，不会误伤合法文件。
const MAX_PACKET: u32 = 16 * 1024 * 1024;

pub(crate) struct Header {
    pub info: FileInfo,
    pub streams: Vec<Stream>,
    /// 按 ASF 流编号（0..128）索引的 payload 扩展定义。
    pub payload_exts: Vec<Vec<PayloadExt>>,
    /// Data Object 的包数（`u64::MAX` = 未知）。
    pub total_packets: u64,
}

pub(crate) fn parse<R: Read>(r: &mut StreamReader<R>) -> Result<Header> {
    let g = r.guid()?;
    if g != guid::HEADER {
        return Err(Error::Bad("missing ASF Header Object"));
    }
    let _header_size = r.u64()?;
    let _num_objects = r.u32()?;
    let _reserved1 = r.u8()?;
    let _reserved2 = r.u8()?;

    let mut info = FileInfo::default();
    let mut streams: Vec<Stream> = Vec::new();
    let mut payload_exts: Vec<Vec<PayloadExt>> = vec![Vec::new(); 128];

    // 主循环一直读到 Data Object 为止 —— 它的包数就是数据区的样本包数。
    let total_packets: u64 = loop {
        let g = r.guid()?;
        let size = r.u64()?;

        if g == guid::DATA {
            // Data Object: File ID(16) + Total Data Packets(8) + Reserved(2)，然后才是数据包。
            let _file_id = r.guid()?;
            let packets = r.u64()?;
            let _reserved = r.u16()?;
            break packets;
        }

        if size < 24 {
            return Err(Error::Bad("object size < 24"));
        }
        let payload_len = (size - 24) as usize;
        if payload_len > MAX_OBJECT {
            return Err(Error::Bad("object payload too large"));
        }
        let payload = r.vec(payload_len)?;
        let mut c = Cursor::new(&payload);

        match g {
            guid::FILE_PROPERTIES => parse_file_properties(&mut c, &mut info)?,
            guid::STREAM_PROPERTIES => {
                if let Some(s) = parse_stream_properties(&mut c)? {
                    streams.push(s);
                }
            }
            guid::EXTENDED_STREAM_PROPERTIES => {
                parse_ext_stream_properties(&mut c, &mut payload_exts)?;
            }
            guid::HEADER_EXTENSION => {
                parse_header_extension(&mut c, &mut payload_exts)?;
            }
            // 其余对象（内容描述 / 编解码器列表 / 语言 / 元数据 / 标记 / 简单索引…）
            // 对「解出裸 ES」没有用，整块跳过。跳过是安全的：主循环读的是**流**，
            // 不依赖对象内部被消费了多少。
            _ => {}
        }
    };

    Ok(Header {
        info,
        streams,
        payload_exts,
        total_packets,
    })
}

fn parse_file_properties(c: &mut Cursor, info: &mut FileInfo) -> Result<()> {
    let _file_id = c.guid()?;
    info.file_size = c.u64()?;
    info.creation_time = c.u64()?;
    info.packet_count = c.u64()?;
    info.play_duration_100ns = c.u64()?;
    info.send_duration_100ns = c.u64()?;
    info.preroll_ms = c.u32()?;
    let _ignore = c.u32()?;
    info.flags = c.u32()?;
    info.min_packet_size = c.u32()?;
    info.max_packet_size = c.u32()?;
    if info.min_packet_size >= (1 << 29) {
        return Err(Error::Bad("min packet size too large"));
    }
    if info.max_packet_size > MAX_PACKET {
        return Err(Error::Bad("max packet size too large"));
    }
    info.max_bitrate = c.u32()?;
    Ok(())
}

fn parse_stream_properties(c: &mut Cursor) -> Result<Option<Stream>> {
    let stream_type = c.guid()?;
    let kind = match stream_type {
        guid::AUDIO_MEDIA => StreamKind::Audio,
        guid::VIDEO_MEDIA => StreamKind::Video,
        guid::COMMAND_MEDIA => StreamKind::Command,
        guid::JFIF_MEDIA => StreamKind::Jfif,
        // 不认识的流类型（文件传输 / 二进制 / Web 流…）：不是错误，直接忽略。
        _ => return Ok(None),
    };

    let _error_correction_type = c.guid()?;
    let time_offset_100ns = c.u64()?;
    let type_specific_len = c.u32()? as usize;
    let _error_correction_len = c.u32()?;
    let flags = c.u16()?;
    let number = (flags & 0x7f) as u8;
    let _reserved = c.u32()?;

    let ts = c.take(type_specific_len)?;
    let mut sc = Cursor::new(ts);

    let mut stream = Stream {
        number,
        kind,
        time_offset_ms: time_offset_100ns / 10_000,
        audio: None,
        video: None,
    };
    match kind {
        StreamKind::Audio => stream.audio = Some(parse_audio_format(&mut sc)?),
        StreamKind::Video => stream.video = Some(parse_video_format(&mut sc)?),
        _ => {}
    }
    Ok(Some(stream))
}

/// WAVEFORMATEX。前 14 字节是固定字段，之后可能有 `wBitsPerSample` / `cbSize` / 私有数据。
fn parse_audio_format(c: &mut Cursor) -> Result<AudioFormat> {
    let mut f = AudioFormat {
        format_tag: c.u16()?,
        channels: c.u16()?,
        sample_rate: c.u32()?,
        avg_bytes_per_sec: c.u32()?,
        block_align: c.u16()?,
        ..Default::default()
    };
    if c.remaining() >= 2 {
        f.bits_per_sample = c.u16()?;
    }
    if c.remaining() >= 2 {
        let _cb_size = c.u16()?;
    }
    if c.remaining() > 0 {
        f.extra = c.take(c.remaining())?.to_vec();
    }
    Ok(f)
}

/// 视频 Type-Specific Data：11 字节前缀 + BITMAPINFOHEADER（`biSize` 通常 40）+ 私有数据。
fn parse_video_format(c: &mut Cursor) -> Result<VideoFormat> {
    let mut f = VideoFormat {
        encoded_width: c.u32()?,
        encoded_height: c.u32()?,
        ..Default::default()
    };
    let _reserved_flags = c.u8()?;
    let _format_data_size = c.u16()?;
    let bitmap_size = c.u32()?;
    f.width = c.u32()?;
    f.height = c.u32()?;
    let _planes = c.u16()?;
    f.bits_per_pixel = c.u16()?;
    f.fourcc = c.u32()?;
    let _size_image = c.u32()?;
    let _x_pels_per_meter = c.u32()?;
    let _y_pels_per_meter = c.u32()?;
    let _clr_used = c.u32()?;
    let _clr_important = c.u32()?;

    if bitmap_size > 40 {
        let want = (bitmap_size - 40) as usize;
        let n = want.min(c.remaining());
        f.extra = c.take(n)?.to_vec();
    }
    Ok(f)
}

fn parse_ext_stream_properties(
    c: &mut Cursor,
    payload_exts: &mut [Vec<PayloadExt>],
) -> Result<()> {
    let _start_time = c.u64()?;
    let _end_time = c.u64()?;
    let _leak_rate = c.u32()?;
    let _bucket_size = c.u32()?;
    let _init_fullness = c.u32()?;
    let _alt_leak_rate = c.u32()?;
    let _alt_bucket_size = c.u32()?;
    let _alt_init_fullness = c.u32()?;
    let _max_object_size = c.u32()?;
    let _flags = c.u32()?;
    let stream_num = c.u16()? as usize;
    let _stream_language_index = c.u16()?;
    let _avg_frame_time = c.u64()?;
    let stream_name_count = c.u16()? as usize;
    let payload_ext_count = c.u16()? as usize;

    for _ in 0..stream_name_count {
        let _lang_index = c.u16()?;
        let len = c.u16()? as usize;
        c.skip(len)?;
    }
    for _ in 0..payload_ext_count {
        let g = c.guid()?;
        let size = c.u16()?;
        let ext_len = c.u32()? as usize;
        c.skip(ext_len)?;
        if stream_num < payload_exts.len() && payload_exts[stream_num].len() < 8 {
            payload_exts[stream_num].push(PayloadExt {
                type_byte: g[0],
                size,
            });
        }
    }
    Ok(())
}

/// Header Extension Object：`Reserved GUID(16) + Reserved(u16) + DataSize(u32) + 子对象…`。
fn parse_header_extension(c: &mut Cursor, payload_exts: &mut [Vec<PayloadExt>]) -> Result<()> {
    let _reserved_guid = c.guid()?;
    let _reserved = c.u16()?;
    let ext_size = c.u32()? as usize;
    let ext = c.take(ext_size)?;

    let mut ec = Cursor::new(ext);
    while ec.remaining() >= 24 {
        let g = ec.guid()?;
        let size = ec.u64()?;
        if size < 24 {
            break;
        }
        let plen = (size - 24) as usize;
        if plen > ec.remaining() {
            break;
        }
        let payload = ec.take(plen)?;
        if g == guid::EXTENDED_STREAM_PROPERTIES {
            let mut pc = Cursor::new(payload);
            parse_ext_stream_properties(&mut pc, payload_exts)?;
        }
    }
    Ok(())
}
