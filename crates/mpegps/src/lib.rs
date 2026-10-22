//! MPEG-1 / MPEG-2 **program stream**（`.mpg` / `.mpeg` / `.vob`）解封装。
//!
//! 只做拆包：把 pack / PES 的壳剥掉，交出每个流的 payload，与 `container` 的
//! `Sample` 对齐，上层解码器不必知道数据是从 PS 还是 ASF 来的。
//!
//! ```text
//!   .mpg ──► PsDemuxer ──► Sample { stream, key_frame, timestamp_ms, data }
//! ```
//!
//! 结构（ISO/IEC 11172-1 / 13818-1）：
//!
//! ```text
//!   pack_header  00 00 01 BA  …  每 ~2 KiB 一个，只带 SCR，不切流
//!   system_header 00 00 01 BB …  可选
//!   PES packet   00 00 01 <id> …  id E0..EF 视频，C0..DF 音频，BD 私有
//!   end code     00 00 01 B9
//! ```
//!
//! 关键点：**一个 PES packet 不保证对齐一帧**。视频的一个 PES 常含多帧，
//! 也可能在帧中间被切断，所以这里只保证「按流把 payload 顺序拼起来是对的」，
//! 帧边界由视频解码器自己按 start code 找。

/// 一个 PES packet 的 payload。
#[derive(Debug, Clone)]
pub struct PesPacket {
    /// PES 的 `stream_id`：`0xE0..=0xEF` 视频，`0xC0..=0xDF` 音频，`0xBD` 私有流。
    pub stream_id: u8,
    /// 显示时间戳（毫秒）。PES 头里没有 PTS 时为 `None`。
    pub pts_ms: Option<u64>,
    /// payload 的字节范围（相对整个文件的偏移），避免逐包拷贝。
    pub range: core::ops::Range<usize>,
}

impl PesPacket {
    pub fn is_video(&self) -> bool {
        (0xE0..=0xEF).contains(&self.stream_id)
    }

    pub fn is_audio(&self) -> bool {
        (0xC0..=0xDF).contains(&self.stream_id)
    }
}

/// PS 拆包器。持有整个文件，按顺序吐出 PES payload。
pub struct PsDemuxer {
    data: Vec<u8>,
    pos: usize,
}

impl PsDemuxer {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, pos: 0 }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// 下一个 PES packet；遇到结尾码或数据耗尽返回 `None`。
    ///
    /// pack header / system header / padding 会被跳过 —— 它们不含流数据。
    pub fn next_packet(&mut self) -> Option<PesPacket> {
        loop {
            let start = self.find_start_code(self.pos)?;
            self.pos = start + 4;
            let id = self.data[start + 3];

            match id {
                // pack header：MPEG-1 是 8 字节，MPEG-2 是 14 + stuffing。
                // 不解析 SCR，只跳过 —— 我们不做时钟恢复。
                0xBA => {
                    self.skip_pack_header(self.pos);
                }
                // system header：16 位长度，跳过。
                0xBB => {
                    let len = self.read_u16(self.pos)? as usize;
                    self.pos += 2 + len;
                }
                // program stream map / directory / end：跳过。
                0xB9 => return None,
                0xBC | 0xBE | 0xBF => {
                    let len = self.read_u16(self.pos)? as usize;
                    self.pos += 2 + len;
                }
                // PES：0xC0..=0xEF 以及 0xBD（私有流）。
                0xBD | 0xC0..=0xEF => {
                    if let Some(packet) = self.parse_pes(id) {
                        return Some(packet);
                    }
                }
                // 其它（含视频 elementary stream 的 start code）：
                // 说明这里已经不是 PS 层了，往前挪一点继续找。
                _ => {
                    self.pos = start + 3;
                }
            }
        }
    }

    /// 解析当前 PES packet 头，返回 payload 范围。
    fn parse_pes(&mut self, stream_id: u8) -> Option<PesPacket> {
        let len = self.read_u16(self.pos)? as usize;
        let header_end = if len == 0 {
            // 视频流允许 0 长度，表示「直到下一个 start code」。此时 payload
            // 一直延伸到下一个 `00 00 01 <id>`。
            self.find_next_pes_start(self.pos + 2)?
        } else {
            self.pos + 2 + len
        };

        let mut pts_ms = None;
        // MPEG-2 PES：`10` 标志位 + 7 位标志 + 1 字节头长。
        // MPEG-1 PES：直接是 stuffing 长度 + 标志，没有 `10` 前缀。
        let first = *self.data.get(self.pos + 2)?;
        let payload_start = if first & 0xC0 == 0x80 {
            // MPEG-2
            let flags = *self.data.get(self.pos + 3)?;
            let header_len = *self.data.get(self.pos + 4)? as usize;
            let payload = self.pos + 5 + header_len;
            if flags & 0x80 != 0 {
                pts_ms = self.read_pts(self.pos + 5);
            }
            payload
        } else {
            // MPEG-1：可选 stuffing（0xFF 填充）之后，**第一个字节本身就是**
            // PTS/DTS 的前缀字节 —— `0010` = 只有 PTS，`0011` = PTS + DTS。
            // 它不是一个「标志字节 + 5 字节时间戳」，前缀就在时间戳里。
            let mut p = self.pos + 2;
            while *self.data.get(p)? == 0xFF {
                p += 1;
            }
            match *self.data.get(p)? >> 4 {
                0x2 => {
                    pts_ms = self.read_pts(p);
                    p += 5;
                }
                0x3 => {
                    pts_ms = self.read_pts(p);
                    p += 10;
                }
                // 没有时间戳：这一字节是 stuffing 长度，跳过。
                _ => p += 1,
            }
            p
        };

        let range = payload_start.min(header_end)..header_end.max(payload_start).min(self.data.len());
        self.pos = header_end.max(payload_start);
        Some(PesPacket {
            stream_id,
            pts_ms,
            range,
        })
    }

    /// 33 位 PTS（90 kHz）→ 毫秒。
    fn read_pts(&self, at: usize) -> Option<u64> {
        let b = self.data.get(at..at + 5)?;
        let pts = ((b[0] as u64 & 0x0E) << 29)
            | ((b[1] as u64) << 22)
            | ((b[2] as u64 & 0xFE) << 14)
            | ((b[3] as u64) << 7)
            | ((b[4] as u64) >> 1);
        Some(pts / 90)
    }

    /// MPEG-1 的 pack header 是 8 字节；MPEG-2 的带 3 位 stuffing 计数。
    fn skip_pack_header(&mut self, at: usize) {
        let Some(&b) = self.data.get(at) else {
            self.pos = self.data.len();
            return;
        };
        // 两种 pack header 的长度不同，且 `at` 指向 start code **之后**：
        //   MPEG-2：10 字节，最后一个字节的低 3 位是 pack_stuffing_length；
        //   MPEG-1：8 字节，没有 stuffing 字段。
        if b & 0xC0 == 0x40 {
            let stuffing = self.data.get(at + 9).map_or(0, |b| (b & 0x07) as usize);
            self.pos = at + 10 + stuffing;
        } else {
            self.pos = at + 8;
        }
    }

    fn find_next_pes_start(&self, from: usize) -> Option<usize> {
        let mut i = from;
        while i + 3 < self.data.len() {
            if self.data[i] == 0 && self.data[i + 1] == 0 && self.data[i + 2] == 1 {
                let id = self.data[i + 3];
                if id == 0xBD || (0xC0..=0xEF).contains(&id) {
                    return Some(i);
                }
            }
            i += 1;
        }
        Some(self.data.len())
    }

    fn find_start_code(&self, from: usize) -> Option<usize> {
        let mut i = from;
        while i + 3 < self.data.len() {
            if self.data[i] == 0 && self.data[i + 1] == 0 && self.data[i + 2] == 1 {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    fn read_u16(&self, at: usize) -> Option<u16> {
        let b = self.data.get(at..at + 2)?;
        Some(u16::from_be_bytes([b[0], b[1]]))
    }
}

/// 从 PES payload 判断这是不是一个 I 帧的起点。
///
/// MPEG-1 和 MPEG-2 的 picture header 布局相同：`00 00 01 00` 之后 2 字节是
/// 10 位 `temporal_reference` + 3 位 `picture_coding_type`，I 帧为 1。
pub fn starts_with_i_picture(bytes: &[u8]) -> bool {
    picture_coding_type(bytes) == Some(1)
}

/// payload 里第一个 picture header 的编码类型（1=I，2=P，3=B）。
pub fn picture_coding_type(bytes: &[u8]) -> Option<u8> {
    let mut i = 0;
    while i + 6 <= bytes.len() {
        if bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 && bytes[i + 3] == 0x00 {
            return Some((bytes[i + 5] >> 3) & 0x07);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一个最小的 MPEG-2 pack header + 一个视频 PES。
    fn pack_with_video(payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        // pack header（MPEG-2，无 stuffing）：00 00 01 BA + 10 字节
        out.extend_from_slice(&[0, 0, 1, 0xBA, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        // PES：00 00 01 E0 + 长度 + MPEG-2 头（3 字节）+ payload
        out.extend_from_slice(&[0, 0, 1, 0xE0]);
        let len = 3 + payload.len();
        out.extend_from_slice(&(len as u16).to_be_bytes());
        out.extend_from_slice(&[0x80, 0x00, 0x00]);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn extracts_the_payload_of_a_video_pes() {
        let payload = [0u8, 0, 1, 0xB3, 0xAA, 0xBB];
        let file = pack_with_video(&payload);
        let mut demux = PsDemuxer::new(file);

        let packet = demux.next_packet().expect("a packet");
        assert!(packet.is_video());
        assert_eq!(&demux.data()[packet.range.clone()], &payload);
        assert!(demux.next_packet().is_none());
    }

    #[test]
    fn skips_the_pack_header_and_finds_the_pes() {
        let file = pack_with_video(&[1, 2, 3]);
        let mut demux = PsDemuxer::new(file);
        // 第一个包必须是 PES，不是 pack header。
        let packet = demux.next_packet().unwrap();
        assert_eq!(packet.stream_id, 0xE0);
        // pack header 14 字节 + PES start code 4 + 长度 2 + MPEG-2 头 3
        assert_eq!(packet.range.start, 23);
    }

    #[test]
    fn reads_a_pts_from_the_pes_header() {
        // PTS = 90000（1 秒）→ 1000 ms
        let pts: u64 = 90000;
        let mut hdr = [0u8; 5];
        hdr[0] = 0x21 | (((pts >> 30) as u8) & 0x0E);
        hdr[1] = (pts >> 22) as u8;
        hdr[2] = 0x01 | (((pts >> 14) as u8) & 0xFE);
        hdr[3] = (pts >> 7) as u8;
        hdr[4] = 0x01 | (((pts << 1) as u8) & 0xFE);

        let mut file = Vec::new();
        file.extend_from_slice(&[0, 0, 1, 0xE0]);
        file.extend_from_slice(&(3u16 + 5 + 1).to_be_bytes());
        file.extend_from_slice(&[0x80, 0x80, 0x05]);
        file.extend_from_slice(&hdr);
        file.push(0xAA);

        let mut demux = PsDemuxer::new(file);
        let packet = demux.next_packet().unwrap();
        assert_eq!(packet.pts_ms, Some(1000));
        assert_eq!(demux.data()[packet.range.clone()], [0xAA]);
    }

    #[test]
    fn recognises_an_i_picture_start() {
        // picture_coding_type = 1 放在第 6 个字节的高 3 位
        let mut payload = vec![0u8, 0, 1, 0x00, 0x00, 0x08];
        payload.extend_from_slice(&[0xFF; 8]);
        assert_eq!(picture_coding_type(&payload), Some(1));
        assert!(starts_with_i_picture(&payload));

        // 把类型改成 3（B 帧）
        let mut b = payload.clone();
        b[5] = 0x18;
        assert_eq!(picture_coding_type(&b), Some(3));
        assert!(!starts_with_i_picture(&b));
    }
}
