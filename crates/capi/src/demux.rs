//! 容器层：嗅探 + 解封装（`v2v_probe` / `v2v_demux_*`）。
//!
//! ## 为什么 ASF 与 AVI 共用一套符号
//!
//! 两个解封装器的动词本来就一样：`new(reader)` / `streams()` / `next_sample()`，
//! 吐出来的也是同一个 `Sample`（`v2v-container` 的类型）。所以这一层不是
//! 「把两种容器塞进一个抽象」，而是**把已经相同的形状照原样搬出去**。
//!
//! ## 样本是借用的
//!
//! `v2v_demux_next` 给出的 `data` 指向句柄内部，**活到下一次 `next` / `close`**。
//! 想留就自己拷 —— 这里刻意不做「交所有权 + free」，因为调用方的典型动作是
//! 「拿到就解」，多一次分配纯属浪费。

use std::fs::File;
use std::io::{BufReader, Read};
use std::os::raw::c_int;
use std::path::Path;

use v2v_asf::{AsfDemuxer, SimpleIndex};
use v2v_avi::AviDemuxer;
use v2v_container::{AudioFormat, Sample, Stream, StreamKind, VideoFormat};

use crate::common::*;

/// 认不出来的容器。
pub const CONTAINER_UNKNOWN: c_int = 0;
/// ASF（`.wmv` / `.wma`）。
pub const CONTAINER_ASF: c_int = 1;
/// AVI（RIFF）。
pub const CONTAINER_AVI: c_int = 2;
/// MPEG-1/2 program stream（`.mpg` / `.dat` / `.vob`）。
///
/// 本仓的解封装器是「整块内存 + `next_packet`」，与这里的流式接口形状不同，
/// 所以**它不在这套符号里**（见 `v2v_mpeg_*`）。
pub const CONTAINER_MPEG_PS: c_int = 3;

/// 流的种类。
pub const STREAM_AUDIO: u32 = 0;
pub const STREAM_VIDEO: u32 = 1;
pub const STREAM_COMMAND: u32 = 2;
pub const STREAM_JFIF: u32 = 3;
pub const STREAM_UNKNOWN: u32 = 4;

/// 一条流的静态信息。
///
/// 视频字段与音频字段在同一块里（没有音频轨时音频字段为 0，反之亦然）。
/// `extra` 是编解码器私有数据，**活到本句柄的下一次 `v2v_demux_stream` / `close`**。
#[repr(C)]
pub struct V2vStreamInfo {
    pub size: u32,
    /// 容器内的流号。**只在同一容器内可比**（ASF 1 起、AVI 0 起）。
    pub number: u32,
    /// `V2V_STREAM_*`。
    pub kind: u32,
    pub time_offset_ms: u64,

    // ---- 视频 ----
    pub width: u32,
    pub height: u32,
    /// 编码器声明里的宽高（可能与显示尺寸不同）。
    pub encoded_width: u32,
    pub encoded_height: u32,
    /// `biCompression`，也就是 FourCC。
    pub fourcc: u32,
    pub bits_per_pixel: u32,

    // ---- 音频 ----
    /// `wFormatTag`：`0x0161` = WMA v2、`0x0160` = WMA v1、`0x0001` = PCM…
    pub format_tag: u32,
    pub sample_rate: u32,
    pub avg_bytes_per_sec: u32,
    pub channels: u32,
    pub block_align: u32,
    pub bits_per_sample: u32,

    /// 编解码器私有数据（借用的，见结构体注释）。
    pub extra: *const u8,
    pub extra_len: u32,
}

/// 一个已重组的样本（视频一帧 / 音频一帧）。
#[repr(C)]
pub struct V2vSample {
    pub size: u32,
    /// 与 `v2v_stream_info.number` 对应。
    pub stream: u32,
    pub key_frame: i32,
    /// 毫秒。ASF 已减掉 preroll；AVI 从 0 起。
    pub timestamp_ms: i64,
    /// 借用的字节，活到本句柄的下一次 `v2v_demux_next` / `close`。
    pub data: *const u8,
    pub data_len: u32,
}

/// 索引（只有 ASF 有）。
///
/// 槽 `i` 的语义：**seek 到第 `i` 槽该从哪个包起读**（是「不晚于它」的那个关键帧）。
/// 所以 `slot = ms / interval_ms`，然后 `v2v_demux_seek_packet(packets[slot])`。
#[repr(C)]
pub struct V2vIndex {
    pub size: u32,
    /// 每槽覆盖的毫秒数（通常 1000）。
    pub interval_ms: u32,
    pub count: u32,
    /// 借用的 `count` 个包号，活到本句柄的下一次 `v2v_demux_index` / `close`。
    pub packets: *const u32,
}

/// 两个解封装器的错误类型不同（`asf::Error` / `avi::Error`），这里收口成
/// 「种类 + 消息」，免得每处 `match` 都要写两条错误分支。
type DemuxResult<T> = Result<T, (c_int, String)>;

fn asf_r<T>(r: Result<T, v2v_asf::Error>) -> DemuxResult<T> {
    r.map_err(|e| (e.kind(), e.message()))
}

fn avi_r<T>(r: Result<T, v2v_avi::Error>) -> DemuxResult<T> {
    r.map_err(|e| (e.kind(), e.message()))
}

enum Backend {
    Asf(AsfDemuxer<BufReader<File>>),
    Avi(AviDemuxer<BufReader<File>>),
}

/// 解封装句柄（不透明）。
pub struct V2vDemux {
    backend: Backend,
    /// 最近一次交出去的样本 —— 出参里的 `data` 指向这里。
    sample: Option<Sample>,
    /// 最近一次 `stream()` 的 extra 拷贝。
    extra: Vec<u8>,
    /// 懒加载的索引（外层 `None` = 还没读；内层 `None` = 文件里没有）。
    index: Option<Option<SimpleIndex>>,
    /// 索引里的包号拷贝（`V2vIndex.packets` 指向这里）。
    packets: Vec<u32>,
}

fn kind_of(k: StreamKind) -> u32 {
    match k {
        StreamKind::Audio => STREAM_AUDIO,
        StreamKind::Video => STREAM_VIDEO,
        StreamKind::Command => STREAM_COMMAND,
        StreamKind::Jfif => STREAM_JFIF,
        StreamKind::Unknown => STREAM_UNKNOWN,
    }
}

fn fill_stream(stream: &Stream, extra: &mut Vec<u8>) -> V2vStreamInfo {
    let v: Option<&VideoFormat> = stream.video.as_ref();
    let a: Option<&AudioFormat> = stream.audio.as_ref();
    extra.clear();
    if let Some(v) = v {
        extra.extend_from_slice(&v.extra);
    } else if let Some(a) = a {
        extra.extend_from_slice(&a.extra);
    }
    V2vStreamInfo {
        size: std::mem::size_of::<V2vStreamInfo>() as u32,
        number: u32::from(stream.number),
        kind: kind_of(stream.kind),
        time_offset_ms: stream.time_offset_ms,
        width: v.map_or(0, |x| x.width),
        height: v.map_or(0, |x| x.height),
        encoded_width: v.map_or(0, |x| x.encoded_width),
        encoded_height: v.map_or(0, |x| x.encoded_height),
        fourcc: v.map_or(0, |x| x.fourcc),
        bits_per_pixel: v.map_or(0, |x| u32::from(x.bits_per_pixel)),
        format_tag: a.map_or(0, |x| u32::from(x.format_tag)),
        sample_rate: a.map_or(0, |x| x.sample_rate),
        avg_bytes_per_sec: a.map_or(0, |x| x.avg_bytes_per_sec),
        channels: a.map_or(0, |x| u32::from(x.channels)),
        block_align: a.map_or(0, |x| u32::from(x.block_align)),
        bits_per_sample: a.map_or(0, |x| u32::from(x.bits_per_sample)),
        extra: extra.as_ptr(),
        extra_len: extra.len() as u32,
    }
}

/// 嗅探容器种类（`V2V_CONTAINER_*`）。只看开头几十个字节。
///
/// 认不出来返回 `V2V_CONTAINER_UNKNOWN`（不是错误）。
///
/// # Safety
/// `head` 要么为空指针（此时按 `len == 0` 处理），要么指向 `len` 字节可读内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_probe(head: *const u8, len: u32) -> c_int {
    guard(|| {
        let b = match unsafe { slice(head, len) } {
            Ok(b) => b,
            Err(e) => {
                set_err(KIND_BAD, &e);
                return CONTAINER_UNKNOWN;
            }
        };
        if v2v_asf::is_asf(b) {
            CONTAINER_ASF
        } else if v2v_avi::is_avi(b) {
            CONTAINER_AVI
        } else if b.len() >= 4 && b[..4] == [0x00, 0x00, 0x01, 0xBA] {
            CONTAINER_MPEG_PS
        } else {
            CONTAINER_UNKNOWN
        }
    })
}

/// 打开一个文件（自动嗅探容器）。失败返回空指针。
///
/// # Safety
/// `path` 必须是 NUL 结尾的 UTF-8 字符串。
#[no_mangle]
pub unsafe extern "C" fn v2v_demux_open(path: *const std::ffi::c_char) -> *mut V2vDemux {
    guard_ptr(|| {
        let p = match unsafe { cstr(path, "path") } {
            Ok(s) => s,
            Err(e) => {
                set_err(KIND_BAD, &e);
                return std::ptr::null_mut();
            }
        };
        let path = Path::new(p);

        // 先读开头几个字节认容器，再按容器开对应的解封装器。
        let kind = match sniff(path) {
            Ok(k) => k,
            Err((k, m)) => {
                set_err(k, &m);
                return std::ptr::null_mut();
            }
        };
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                set_err(KIND_IO, &format!("打不开 {}: {e}", path.display()));
                return std::ptr::null_mut();
            }
        };
        let reader = BufReader::new(file);
        let backend = match kind {
            CONTAINER_ASF => match AsfDemuxer::new(reader) {
                Ok(d) => Backend::Asf(d),
                Err(e) => {
                    rc(&e);
                    return std::ptr::null_mut();
                }
            },
            CONTAINER_AVI => match AviDemuxer::new(reader) {
                Ok(d) => Backend::Avi(d),
                Err(e) => {
                    rc(&e);
                    return std::ptr::null_mut();
                }
            },
            _ => {
                set_err(
                    KIND_UNSUPPORTED,
                    "这个容器不在这套符号里（MPEG-PS 走 v2v_mpeg_*）",
                );
                return std::ptr::null_mut();
            }
        };
        Box::into_raw(Box::new(V2vDemux {
            backend,
            sample: None,
            extra: Vec::new(),
            index: None,
            packets: Vec::new(),
        }))
    })
}

fn sniff(path: &Path) -> Result<c_int, (c_int, String)> {
    let mut f = File::open(path).map_err(|e| (KIND_IO, format!("打不开 {}: {e}", path.display())))?;
    let mut head = [0u8; 64];
    let n = f
        .read(&mut head)
        .map_err(|e| (KIND_IO, format!("读 {} 失败: {e}", path.display())))?;
    let b = &head[..n];
    let k = if v2v_asf::is_asf(b) {
        CONTAINER_ASF
    } else if v2v_avi::is_avi(b) {
        CONTAINER_AVI
    } else {
        CONTAINER_UNKNOWN
    };
    if k == CONTAINER_UNKNOWN {
        return Err((KIND_UNSUPPORTED, "认不出这个容器".into()));
    }
    Ok(k)
}

/// 关掉并释放。传空指针是空操作。
///
/// # Safety
/// `h` 必须是 [`v2v_demux_open`] 的返回值，且只能关一次。
#[no_mangle]
pub unsafe extern "C" fn v2v_demux_close(h: *mut V2vDemux) {
    if h.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(h) });
}

/// 有几条流。
///
/// # Safety
/// `h` 必须是有效句柄。
#[no_mangle]
pub unsafe extern "C" fn v2v_demux_stream_count(h: *const V2vDemux) -> u32 {
    if h.is_null() {
        return 0;
    }
    let hp = unsafe { &*h };
    match &hp.backend {
        Backend::Asf(d) => d.streams().len() as u32,
        Backend::Avi(d) => d.streams().len() as u32,
    }
}

/// 取第 `i` 条流的信息。`i` 越界返回 `V2V_ERR_ARG`。
///
/// `out->extra` 活到本句柄的下一次 `v2v_demux_stream` / `close`。
///
/// # Safety
/// `out` 必须指向一块首字段是 `size` 的可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_demux_stream(
    h: *mut V2vDemux,
    i: u32,
    out: *mut V2vStreamInfo,
) -> c_int {
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        let hp = unsafe { &mut *h };
        let streams: &[Stream] = match &hp.backend {
            Backend::Asf(d) => d.streams(),
            Backend::Avi(d) => d.streams(),
        };
        let Some(s) = streams.get(i as usize) else {
            return arg_err(&format!("流下标 {i} 越界（共 {} 条）", streams.len()));
        };
        // 先把借用结束掉，再改 `hp.extra`。
        let info = fill_stream(s, &mut hp.extra);
        unsafe { write_sized(out, info) };
        OK
    })
}

/// 取下一个样本。[`EOS`] = 到头。
///
/// `out->data` 活到本句柄的下一次 `v2v_demux_next` / `close`。
///
/// # Safety
/// `out` 必须指向一块首字段是 `size` 的可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_demux_next(h: *mut V2vDemux, out: *mut V2vSample) -> c_int {
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        let hp = unsafe { &mut *h };
        let next = match &mut hp.backend {
            Backend::Asf(d) => asf_r(d.next_sample()),
            Backend::Avi(d) => avi_r(d.next_sample()),
        };
        match next {
            Ok(None) => EOS,
            Ok(Some(s)) => {
                hp.sample = Some(s);
                let s = hp.sample.as_ref().expect("刚放进去");
                let v = V2vSample {
                    size: std::mem::size_of::<V2vSample>() as u32,
                    stream: u32::from(s.stream),
                    key_frame: i32::from(s.key_frame),
                    timestamp_ms: s.timestamp_ms,
                    data: s.data.as_ptr(),
                    data_len: s.data.len() as u32,
                };
                unsafe { write_sized(out, v) };
                OK
            }
            Err((k, m)) => rc_kind(k, &m),
        }
    })
}

/// 把读游标挪到第 `packet` 个包。之后用 `next` 往后读。
///
/// 只有 ASF 有索引，AVI 返回 `V2V_ERR_UNSUPPORTED`。
///
/// # Safety
/// `h` 必须是有效句柄。
#[no_mangle]
pub unsafe extern "C" fn v2v_demux_seek_packet(h: *mut V2vDemux, packet: u64) -> c_int {
    guard(|| {
        if h.is_null() {
            return arg_err("句柄是空指针");
        }
        let hp = unsafe { &mut *h };
        match &mut hp.backend {
            Backend::Asf(d) => match d.seek_to_packet(packet) {
                Ok(()) => OK,
                Err(e) => rc(&e),
            },
            Backend::Avi(_) => {
                set_err(KIND_UNSUPPORTED, "AVI 没有包级索引，不支持按包 seek");
                ERR_UNSUPPORTED
            }
        }
    })
}

/// 取索引。[`EOS`] = 这个文件没有索引（或容器不支持索引）。
///
/// `out->packets` 活到本句柄的下一次 `v2v_demux_index` / `close`。
///
/// # Safety
/// `out` 必须指向一块首字段是 `size` 的可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_demux_index(h: *mut V2vDemux, out: *mut V2vIndex) -> c_int {
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        let hp = unsafe { &mut *h };
        if hp.index.is_none() {
            let got = match &mut hp.backend {
                Backend::Asf(d) => d.simple_index(),
                Backend::Avi(_) => Ok(None),
            };
            match got {
                Ok(i) => hp.index = Some(i),
                Err(e) => return rc(&e),
            }
        }
        let Some(Some(idx)) = &hp.index else {
            return EOS;
        };
        hp.packets.clear();
        hp.packets.extend_from_slice(&idx.packet_numbers);
        let v = V2vIndex {
            size: std::mem::size_of::<V2vIndex>() as u32,
            interval_ms: (idx.interval_100ns / 10_000).max(1) as u32,
            count: hp.packets.len() as u32,
            packets: hp.packets.as_ptr(),
        };
        unsafe { write_sized(out, v) };
        OK
    })
}
