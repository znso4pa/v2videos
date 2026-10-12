//! 纯 Rust 的 ASF（`.wmv` / `.wma`）解封装。
//!
//! 目标：把 ASF 容器拆开，逐条吐出**裸基本流样本**（视频一帧 / 音频一帧，带时间戳与
//! 关键帧标志）。后面接解码器（MS-MPEG-4 / WMA…）就得到 YUV / PCM。
//!
//! ```text
//!   .wmv/.wma ──► [本 crate] ──► Sample { stream, key_frame, timestamp_ms, data }
//!                  解封装
//!
//! 反方向（封装）也有：`AsfMuxer` 把裸样本写回 `.wmv`（目前单视频流）。
//! ```
//!
//! 无 C、无 FFI、零依赖。数据包布局按 ASF 规范实现；行为与 `ffprobe` 交叉验证
//! （见 `tests/oracle.rs`：轨道、尺寸、包数、关键帧数逐字段对）。
//!
//! ## 用法
//!
//! ```no_run
//! # fn main() -> Result<(), v2v_asf::Error> {
//! let file = std::fs::File::open("sample.wmv")?;
//! let mut d = v2v_asf::AsfDemuxer::new(file)?;
//! for s in d.streams() {
//!     println!("stream {} {:?}", s.number, s.kind);
//! }
//! while let Some(sample) = d.next_sample()? {
//!     let _ = sample.data.len();
//! }
//! # Ok(()) }
//! ```

mod cursor;
mod error;
mod guid;
mod header;
mod mux;
mod types;

pub use error::{Error, Result};
pub use mux::{AsfMuxer, VideoStreamInfo};
pub use types::{
    fourcc_string, AudioFormat, FileInfo, Sample, Stream, StreamKind, VideoFormat,
};

use std::collections::VecDeque;
use std::io::Read;

use cursor::{Cursor, StreamReader};
use types::PayloadExt;

/// 一帧 payload 头的最小长度。
const FRAME_HEADER_SIZE: i64 = 6;

/// 每条流的重组状态。
struct StreamState {
    /// 正在拼装的样本缓冲（长度 = `packet_obj_size`）。
    buf: Vec<u8>,
    frag_offset: usize,
    timestamp_ms: i64,
    key_frame: bool,
    payload_exts: Vec<PayloadExt>,
}

/// ASF 解封装器。
pub struct AsfDemuxer<R: Read> {
    reader: StreamReader<R>,
    info: FileInfo,
    streams: Vec<Stream>,
    states: Vec<StreamState>,
    /// 按流编号标记该流是否在 Stream Properties 里声明过。
    known: Vec<bool>,
    packet_size: usize,
    min_packet_size: u32,
    preroll_ms: u32,
    total_packets: u64,
    packets_read: u64,
    queue: VecDeque<Sample>,
    done: bool,
}

impl<R: Read> AsfDemuxer<R> {
    /// 解析 Header Object，准备读数据包。
    pub fn new(reader: R) -> Result<Self> {
        let mut reader = StreamReader::new(reader);
        let h = header::parse(&mut reader)?;

        let mut known = vec![false; 128];
        for s in &h.streams {
            if (s.number as usize) < known.len() {
                known[s.number as usize] = true;
            }
        }
        let states = h
            .payload_exts
            .iter()
            .map(|e| StreamState {
                buf: Vec::new(),
                frag_offset: 0,
                timestamp_ms: 0,
                key_frame: false,
                payload_exts: e.clone(),
            })
            .collect();

        Ok(Self {
            reader,
            packet_size: h.info.max_packet_size.max(1) as usize,
            min_packet_size: h.info.min_packet_size,
            preroll_ms: h.info.preroll_ms,
            total_packets: h.total_packets,
            info: h.info,
            streams: h.streams,
            states,
            known,
            packets_read: 0,
            queue: VecDeque::new(),
            done: false,
        })
    }

    pub fn info(&self) -> &FileInfo {
        &self.info
    }

    pub fn streams(&self) -> &[Stream] {
        &self.streams
    }

    /// 下一条完整样本；`None` = 到文件尾。
    pub fn next_sample(&mut self) -> Result<Option<Sample>> {
        loop {
            if let Some(s) = self.queue.pop_front() {
                return Ok(Some(s));
            }
            if self.done {
                return Ok(None);
            }
            self.read_one_packet()?;
        }
    }

    fn is_audio(&self, stream: usize) -> bool {
        self.streams
            .iter()
            .any(|s| s.number as usize == stream && s.kind == StreamKind::Audio)
    }

    /// 读一个 ASF 数据包并把它内部的 payload 全部处理进队列。
    fn read_one_packet(&mut self) -> Result<()> {
        if self.packets_read >= self.total_packets {
            self.done = true;
            return Ok(());
        }
        let mut buf = vec![0u8; self.packet_size];
        let filled = self.reader.read_full(&mut buf)?;
        if filled < self.packet_size {
            self.done = true;
            if filled == 0 {
                return Ok(());
            }
            buf.truncate(filled);
        }
        self.packets_read += 1;
        self.process_packet(&buf)
    }

    fn process_packet(&mut self, buf: &[u8]) -> Result<()> {
        let preroll = self.preroll_ms as i64;
        let mut c = Cursor::new(buf);

        // ---- Error Correction Flags ----
        let ec = c.u8()?;
        if ec & 0x80 != 0 {
            if ec & 0x60 != 0 {
                return Err(Error::Unsupported(
                    "non-standard ASF error correction flags".into(),
                ));
            }
            let ec_len = (ec & 0x0f) as usize;
            if ec_len < 2 {
                return Err(Error::Bad("error correction data length < 2"));
            }
            c.skip(ec_len)?;
        }

        let packet_flags = c.u8()?;
        let packet_property = c.u8()?;

        let packet_length = read_2bits(&mut c, packet_flags >> 5)?
            .unwrap_or(self.packet_size as u32);
        let _sequence = read_2bits(&mut c, packet_flags >> 1)?;
        let mut padsize = read_2bits(&mut c, packet_flags >> 3)?.unwrap_or(0);

        let packet_timestamp = c.u32()? as i64;
        let _duration = c.u16()?;

        let (mut packet_segments, segsizetype) = if packet_flags & 0x01 != 0 {
            let s = c.u8()?;
            ((s & 0x3f) as i64, s)
        } else {
            (1i64, 0x80u8)
        };

        if packet_length == 0 || packet_length >= (1 << 29) {
            return Err(Error::Bad("invalid packet length"));
        }
        if padsize >= packet_length {
            return Err(Error::Bad("padding >= packet length"));
        }
        let header_len = c.pos();
        if header_len as u32 > packet_length - padsize {
            return Err(Error::Bad("packet header longer than packet"));
        }
        let mut size_left = (packet_length - padsize) as i64 - header_len as i64;
        if packet_length < self.min_packet_size {
            padsize += self.min_packet_size - packet_length;
        }
        let _ = padsize;

        // ---- payloads ----
        let mut time_start: i64 = 0;
        let mut time_delta: u8 = 0;
        let mut multi_size: i64 = 0;

        // 跨迭代保留的当前帧头状态。
        let mut cur_stream: usize = 0;
        let mut cur_key = false;
        let mut cur_replic: usize = 0;
        let mut obj_size: usize = 0;
        let mut have_obj = false;
        let mut cur_ts: i64 = packet_timestamp;

        loop {
            if size_left < FRAME_HEADER_SIZE || (packet_segments < 1 && time_start == 0) {
                break;
            }

            let mut frag_offset: usize;
            let mut frag_size: usize;

            if time_start == 0 {
                let fh_start = c.pos();
                let num = c.u8()?;
                packet_segments -= 1;
                cur_key = (num >> 7) != 0;
                cur_stream = (num & 0x7f) as usize;

                let _seq = read_2bits(&mut c, packet_property >> 4)?;
                frag_offset = read_2bits(&mut c, packet_property >> 2)?.unwrap_or(0) as usize;
                cur_replic = read_2bits(&mut c, packet_property)?.unwrap_or(0) as usize;
                obj_size = 0;
                have_obj = false;

                if cur_replic >= 8 {
                    let end = c.pos() + cur_replic;
                    if end > buf.len() {
                        return Err(Error::Bad("replicated data past packet end"));
                    }
                    obj_size = c.u32()? as usize;
                    cur_ts = c.u32()? as i64;
                    have_obj = true;

                    if cur_stream < 128 {
                        let exts = std::mem::take(&mut self.states[cur_stream].payload_exts);
                        for p in &exts {
                            let mut sz = p.size as usize;
                            if p.size == 0xFFFF {
                                sz = c.u16()? as usize;
                            }
                            let pay_end = c.pos() + sz;
                            if pay_end > end {
                                break;
                            }
                            if p.type_byte == 0x2A {
                                c.skip(8)?;
                                let ts0 = c.u64()?;
                                let _ts1 = c.u64()?;
                                if ts0 != u64::MAX {
                                    cur_ts = (ts0 / 10_000) as i64;
                                }
                            }
                            c.seek(pay_end)?;
                        }
                        self.states[cur_stream].payload_exts = exts;
                    }
                    c.seek(end)?;
                } else if cur_replic == 1 {
                    // 压缩载荷：frag_offset 这个字段在这里当「起始时间戳」用。
                    time_start = frag_offset as i64;
                    frag_offset = 0;
                    cur_ts = packet_timestamp;
                    time_delta = c.u8()?;
                } else if cur_replic != 0 {
                    return Err(Error::Bad("unexpected replicated data length"));
                }

                let rsize = c.pos() - fh_start;
                if rsize as i64 > size_left {
                    break;
                }

                if packet_flags & 0x01 != 0 {
                    let v = read_2bits(&mut c, segsizetype >> 6)?.unwrap_or(0) as usize;
                    if v == 0 {
                        break;
                    }
                    frag_size = v;
                } else {
                    frag_size = (size_left - rsize as i64) as usize;
                }
                if cur_replic == 1 {
                    multi_size = frag_size as i64;
                }
                size_left -= rsize as i64;
            } else {
                // 压缩载荷的后续子载荷：帧头只在第一段读一次。
                frag_offset = 0;
                frag_size = 0;
            }

            // 压缩载荷（replic_size == 1）：每个子载荷自带一字节长度与递推时间戳。
            if cur_replic == 1 {
                cur_ts = time_start;
                time_start += time_delta as i64;
                let sz = c.u8()? as usize;
                obj_size = sz;
                frag_size = sz;
                have_obj = true;
                size_left -= 1;
                multi_size -= 1;
                if multi_size < obj_size as i64 {
                    time_start = 0;
                    let skip = multi_size.max(0) as usize;
                    c.skip(skip.min(c.remaining()))?;
                    size_left -= multi_size;
                    continue;
                }
                multi_size -= obj_size as i64;
            }

            // 未知流：整段跳过。
            if cur_stream >= 128 || !self.known[cur_stream] {
                c.skip(frag_size.min(c.remaining()))?;
                size_left -= frag_size as i64;
                continue;
            }

            if !have_obj {
                obj_size = frag_size;
            }

            // 重组。
            let is_audio = self.is_audio(cur_stream);
            let complete = {
                let st = &mut self.states[cur_stream];
                if st.buf.len() != obj_size || st.frag_offset + frag_size > st.buf.len() {
                    st.buf = vec![0u8; obj_size];
                    st.frag_offset = 0;
                    st.timestamp_ms = cur_ts - preroll;
                    st.key_frame = cur_key;
                }
                if frag_offset >= st.buf.len() || frag_size > st.buf.len() - frag_offset {
                    // 片段位置非法：丢弃这一段，别污染缓冲。
                    None
                } else {
                    let data = c.take(frag_size)?;
                    st.buf[frag_offset..frag_offset + frag_size].copy_from_slice(data);
                    size_left -= frag_size as i64;
                    st.frag_offset = frag_offset + frag_size;
                    if st.frag_offset == st.buf.len() {
                        let sample = Sample {
                            stream: cur_stream as u8,
                            key_frame: st.key_frame || is_audio,
                            timestamp_ms: st.timestamp_ms,
                            data: std::mem::take(&mut st.buf),
                        };
                        st.frag_offset = 0;
                        Some(sample)
                    } else {
                        None
                    }
                }
            };
            if let Some(s) = complete {
                self.queue.push_back(s);
            }
        }

        Ok(())
    }
}

/// ASF 的 2-bit 长度类型：`0` = 无字段（用默认值），`1` = u8，`2` = u16，`3` = u32。
fn read_2bits(c: &mut Cursor, bits: u8) -> Result<Option<u32>> {
    Ok(match bits & 3 {
        3 => Some(c.u32()?),
        2 => Some(c.u16()? as u32),
        1 => Some(c.u8()? as u32),
        _ => None,
    })
}
