//! Ogg 页**写入器** —— 解封装的逆操作。
//!
//! ## 为什么需要它
//!
//! 本仓没有 Vorbis **编码器**，所以「写 `.ogg`」不可能是「PCM → Vorbis」。
//! 能做的是**重封装**：把已经解出来的包按新的分页重新写出去。用途是
//! 无损剪切 / 拼接 —— 这两件事都只搬包、不改码流。
//!
//! ## 页的规矩（踩错就不报错，只是文件坏了）
//!
//! - **段表**：一个包切成「若干个 255 + 一个余数」。包长正好是 255 的整数倍时
//!   **末尾要补一个 0** —— 否则解码器会把下一个包的开头当成本包的续接。
//! - **一页最多 255 个段**（`MAX_BODY = 255 × 255` 字节）。放不下的包要
//!   **跨页续接**：下一页的 `continued` 位置 1，且这一页的 `granule` 写 `-1`
//!   （本页没有包结束）。
//! - **`granule` 只描述本页最后一个完成的包**。一页里有 N 个包时，前面 N−1 个的
//!   结束位置解码器自己往前推。
//! - **CRC 覆盖整页**（含页头与段表），计算时把 CRC 字段本身当 4 个 0。
//! - `serial` 在一份文件里唯一；`sequence` 从 0 起连续。
//!
//! ## 一页放几个包
//!
//! 少了文件变大（每页 27 + 段表 + 最多 4 字节 CRC 的开销），多了 seek 变粗。
//! libvorbis 默认一页约 1 秒音频；这里用**包数**（[`OggMuxer::with_page_packets`]）
//! 而不是时长 —— 因为写入器不该假装懂 Vorbis 的采样率。

use crate::crc::crc32_page;
use crate::page::{CAPTURE, GRANULE_NONE, HEADER_SIZE, MAX_BODY};
use crate::{Error, Result};

/// CRC 字段在页里的偏移：`OggS`(4) + version(1) + flags(1) + granule(8) + serial(4) + seq(4)。
const CRC_OFFSET: usize = 22;

/// 一页默认放多少个包。
///
/// 64 个 Vorbis 包 ≈ 64 × 1024 个采样点 ≈ 1.5 秒（44.1 kHz）—— 与 libvorbis
/// 自己挑的量级一致。太小文件会胖，太大 seek 会粗。
pub const DEFAULT_PAGE_PACKETS: usize = 64;

/// 按页写 Ogg。
///
/// ```no_run
/// use v2v_ogg::OggMuxer;
///
/// let mut m = OggMuxer::new(0x1234);
/// m.write_packet(b"\x01vorbis....", 0).unwrap(); // 识别头
/// m.write_packet(b"\x03vorbis....", 0).unwrap(); // 注释头
/// m.write_packet(b"\x05vorbis....", 0).unwrap(); // setup 头
/// m.write_packet(&[0u8; 100], 1024).unwrap();    // 一个音频包
/// let bytes: Vec<u8> = m.finish();
/// # let _ = bytes;
/// ```
#[derive(Debug)]
pub struct OggMuxer {
    out: Vec<u8>,
    serial: u32,
    sequence: u32,
    /// 一页最多放多少个**完成的**包。
    page_packets: usize,
    /// 当前页的段表。
    segments: Vec<u8>,
    /// 当前页的页体。
    body: Vec<u8>,
    /// 当前页**最后一个完成包**的 granule；本页还没有包结束时是 `GRANULE_NONE`。
    granule: i64,
    /// 当前页里已经完成了几个包。
    packets: usize,
    /// 当前页是不是上一页某个包的续接。
    continued: bool,
    /// 还有没有包没写完（下一页开头要标 `continued`）。
    pending_continue: bool,
    /// 下一页是不是逻辑流的第一页。
    bos: bool,
    /// 是否已经收尾（收尾之后再写就是调用方的错）。
    done: bool,
    /// 最后一页在 `out` 里的起始偏移与长度 —— `finish()` 补 `EOS` 时要改它。
    last_page: Option<(usize, usize)>,
}

impl OggMuxer {
    pub fn new(serial: u32) -> Self {
        Self {
            out: Vec::new(),
            serial,
            sequence: 0,
            page_packets: DEFAULT_PAGE_PACKETS,
            segments: Vec::new(),
            body: Vec::new(),
            granule: GRANULE_NONE,
            packets: 0,
            continued: false,
            pending_continue: false,
            bos: true,
            done: false,
            last_page: None,
        }
    }

    /// 改「一页放几个包」。0 会被抬到 1。
    pub fn with_page_packets(mut self, n: usize) -> Self {
        self.page_packets = n.max(1);
        self
    }

    /// 写一个包。`granule` 是**这个包结束之后**的 PCM 位置（Vorbis 口径 = 采样数）。
    ///
    /// 包头三个（识别 / 注释 / setup）按规范填 0。
    ///
    /// 传 [`GRANULE_NONE`] 表示「本包的位置不知道，别动本页的 granule」——
    /// 规范只要求页 granule 描述**本页最后一个完成的包**，所以重封装时中间那些包
    /// 就该传它（Vorbis 每个包的结束位置不解码是推不出来的）。
    pub fn write_packet(&mut self, data: &[u8], granule: i64) -> Result<()> {
        if self.done {
            return Err(Error::Bad("已经 finish 过了，不能再写".into()));
        }
        // 段表最多 255 项，所以一页最多装 255×255 字节的页体。
        if data.len() > MAX_BODY {
            // 单个包比一页还大：只能跨页。Vorbis 的 setup 头有可能到这个量级。
            return self.write_spanning_packet(data, granule);
        }
        // 放不下就先把本页收掉（不标 EOS）。
        if self.segments.len() + lacing_len(data.len()) > 255
            || self.packets >= self.page_packets
        {
            self.flush_page(false);
        }
        push_lacing(&mut self.segments, data.len());
        self.body.extend_from_slice(data);
        // ⚠️ 只有**最后完成**的那个包才写 granule；`GRANULE_NONE` 表示「不动」。
        if granule != GRANULE_NONE {
            self.granule = granule;
        }
        self.packets += 1;
        // ⚠️ **不在这里 flush**：页满了要留到下一个包进来时才落盘。这样 `finish()`
        // 手里永远握着最后一页，`EOS` 能直接标在它上面 —— 否则就得为 EOS 单独写一个
        // **空页**（实测多出 28 字节、多一页，`ffprobe` 也会多报一页）。
        Ok(())
    }

    /// 超过一页容量的包：按 `MAX_BODY` 切开，每页标续接。
    ///
    /// 规范上只有「页体填满 255 段」才允许跨页，所以这里每页都塞满 255 段
    /// （`255 × 255` 字节），最后一页放余数。
    fn write_spanning_packet(&mut self, data: &[u8], granule: i64) -> Result<()> {
        let mut rest = data;
        while !rest.is_empty() {
            if self.segments.len() == 255 {
                // 本页塞满了还没写完 —— 这一页没有包结束，granule 必须是 -1。
                self.granule = GRANULE_NONE;
                self.flush_page(false);
            }
            let room = 255 - self.segments.len();
            let take = room * 255;
            let n = take.min(rest.len());
            for _ in 0..(n / 255) {
                self.segments.push(255);
            }
            // 余数段（这个包还没结束，但本页要收尾时它一定正好是 255 的倍数，
            // 只有最后一页才可能出现 < 255 的余数）。
            if n % 255 != 0 {
                self.segments.push((n % 255) as u8);
            }
            self.body.extend_from_slice(&rest[..n]);
            rest = &rest[n..];
            if rest.is_empty() {
                self.granule = granule;
                self.packets += 1;
            } else {
                self.pending_continue = true;
                self.granule = GRANULE_NONE;
                self.flush_page(false);
            }
        }
        Ok(())
    }

    /// 把当前页落盘（**不标 EOS**）。
    ///
    /// 需要「保持原文件的分页」时用：每写完一页的包就 `flush()` 一次，
    /// 页边界与 granule 就与原文件一致。
    pub fn flush(&mut self) {
        self.flush_page(false);
    }

    /// 直接写**一整页**：段表、页体、granule、`continued` 都由调用方给。
    ///
    /// 这是重封装（无损剪切 / 拼接）该用的原语 —— 它让产物**保持原文件的分页**。
    /// 用 [`write_packet`](Self::write_packet) 也行，但那会让写入器按自己的规则重新
    /// 分页，于是中间那些页的 `granule` 就成了未知数（Vorbis 每个包的结束位置
    /// 不解码是推不出来的），而「有包结束却写着 `-1`」的页会毁掉 seek。
    ///
    /// `continued` 是页头里那个「本页第一个包是上一页的续接」位。`BOS` 由写入器
    /// 按「是不是第一页」自己标，`EOS` 由 [`finish`](Self::finish) 补。
    pub fn write_page(
        &mut self,
        segments: &[u8],
        body: &[u8],
        granule: i64,
        continued: bool,
    ) -> Result<()> {
        if self.done {
            return Err(Error::Bad("已经 finish 过了，不能再写".into()));
        }
        if segments.len() > 255 {
            return Err(Error::Bad(format!(
                "段表 {} 项，一页最多 255 项",
                segments.len()
            )));
        }
        let expect: usize = segments.iter().map(|&s| s as usize).sum();
        if expect != body.len() {
            return Err(Error::Bad(format!(
                "段表合计 {expect} 字节，页体却是 {} 字节",
                body.len()
            )));
        }
        // 手里还有没落盘的包就先落盘 —— 混用两个 API 时别把数据丢了。
        self.flush_page(false);
        self.segments.extend_from_slice(segments);
        self.body.extend_from_slice(body);
        self.granule = granule;
        self.continued = continued;
        self.flush_page(false);
        Ok(())
    }

    /// 收尾：把剩下的写成一页，标 `EOS`。返回整个文件。
    ///
    /// ⚠️ **手里没有待写的包时，`EOS` 补在最后**已经落盘**的那一页上**
    /// （改标志位 + 重算那一页的 CRC），不另写一个空页 —— 空页既浪费 28 字节，
    /// 也会让 `ffprobe` 多报一页。
    pub fn finish(mut self) -> Vec<u8> {
        if self.done {
            return self.out;
        }
        self.done = true;
        if !self.segments.is_empty() || !self.body.is_empty() {
            self.flush_page(true);
        } else if let Some((at, len)) = self.last_page {
            self.out[at + 5] |= 0x04; // header_type 的 EOS 位
            let crc = crc32_page(&self.out[at..at + len], CRC_OFFSET);
            self.out[at + CRC_OFFSET..at + CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
        }
        self.out
    }

    /// 当前已写出的字节数（还没 flush 的部分不算）。
    pub fn len(&self) -> usize {
        self.out.len()
    }

    pub fn is_empty(&self) -> bool {
        self.out.is_empty()
    }

    fn flush_page(&mut self, eos: bool) {
        // 什么都没攒下（比如一个包都没写过）：不写空页。`EOS` 必须标在**有包**的页上，
        // 而惰性 flush 保证了收尾时最后一页一定还在手里，所以这个守卫是安全的。
        if self.segments.is_empty() && self.body.is_empty() {
            return;
        }
        let mut flags = 0u8;
        if self.continued {
            flags |= 0x01; // continued
        }
        if self.bos {
            flags |= 0x02; // BOS
        }
        if eos {
            flags |= 0x04;
        }

        let mut page = Vec::with_capacity(HEADER_SIZE + self.segments.len() + self.body.len());
        page.extend_from_slice(CAPTURE);
        page.push(0); // stream_structure_version
        page.push(flags);
        page.extend_from_slice(&self.granule.to_le_bytes()); // 小端 i64
        page.extend_from_slice(&self.serial.to_le_bytes());
        page.extend_from_slice(&self.sequence.to_le_bytes());
        page.extend_from_slice(&[0, 0, 0, 0]); // CRC 占位
        page.push(self.segments.len() as u8);
        page.extend_from_slice(&self.segments);
        page.extend_from_slice(&self.body);
        let crc = crc32_page(&page, CRC_OFFSET);
        page[CRC_OFFSET..CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());

        let at = self.out.len();
        let len = page.len();
        self.out.extend_from_slice(&page);
        self.last_page = Some((at, len));
        self.sequence += 1;
        self.bos = false;
        // 下一页开头是不是续接，取决于**本页最后是不是把一个包切断了**。
        self.continued = self.pending_continue;
        self.pending_continue = false;
        self.segments.clear();
        self.body.clear();
        self.granule = GRANULE_NONE;
        self.packets = 0;
    }
}

/// 一个 `len` 字节的包需要几个段。
fn lacing_len(len: usize) -> usize {
    len / 255 + 1
}

/// 把 `len` 字节的包的段表项追加到 `segments`。
///
/// ⚠️ 长度正好是 255 的整数倍时**末尾要补一个 0** —— 规范用「< 255 的段」表示
/// 包结束，少了这个 0，解码器会把下一个包的第一个字节当成本包的续接。
fn push_lacing(segments: &mut Vec<u8>, len: usize) {
    let mut rest = len;
    while rest >= 255 {
        segments.push(255);
        rest -= 255;
    }
    segments.push(rest as u8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::PageReader;

    /// 段表：`255` 的整数倍要补 0，否则包边界就丢了。
    #[test]
    fn lacing_always_terminates_the_packet() {
        let mut s = Vec::new();
        push_lacing(&mut s, 0);
        assert_eq!(s, [0], "空包也要一个 0 段");
        s.clear();
        push_lacing(&mut s, 254);
        assert_eq!(s, [254]);
        s.clear();
        push_lacing(&mut s, 255);
        assert_eq!(s, [255, 0], "正好 255 字节要补一个 0");
        s.clear();
        push_lacing(&mut s, 510);
        assert_eq!(s, [255, 255, 0]);
        s.clear();
        push_lacing(&mut s, 511);
        assert_eq!(s, [255, 255, 1]);
    }

    #[test]
    fn lacing_len_matches_push_lacing() {
        for len in [0usize, 1, 254, 255, 256, 509, 510, 511, 65_025] {
            let mut s = Vec::new();
            push_lacing(&mut s, len);
            assert_eq!(s.len(), lacing_len(len), "len = {len}");
        }
    }

    /// 写完能被自己的解封装器原样读回来（CRC、段表、包内容）。
    #[test]
    fn round_trips_through_our_own_page_reader() {
        let mut m = OggMuxer::new(0xdead_beef).with_page_packets(3);
        let packets: Vec<Vec<u8>> = vec![
            b"id-header".to_vec(),
            b"comment-header".to_vec(),
            vec![7u8; 300],  // 跨两个段
            vec![9u8; 255],  // 正好一个段（要补 0）
            vec![0u8; 0],    // 空包
            vec![3u8; 1000], // 4 个段
        ];
        for (i, p) in packets.iter().enumerate() {
            m.write_packet(p, (i as i64 + 1) * 1024).unwrap();
        }
        let bytes = m.finish();
        assert_eq!(&bytes[..4], CAPTURE);

        let mut r = PageReader::new(&bytes[..]);
        // ⚠️ 默认**不校验** CRC，不打开这一行下面那条 `crc_errors` 就是空断言。
        r.set_verify_crc(true);
        let mut got: Vec<Vec<u8>> = Vec::new();
        let mut last_granule = -1i64;
        let mut seq = 0u32;
        while let Some(page) = r.next_page().unwrap() {
            assert_eq!(page.serial, 0xdead_beef);
            assert_eq!(page.sequence, seq, "页序号必须连续");
            seq += 1;
            if page.granule != GRANULE_NONE {
                last_granule = page.granule;
            }
            // 页内的段拼回包。
            let mut at = 0usize;
            let mut cur: Vec<u8> = Vec::new();
            for &seg in &page.segments {
                cur.extend_from_slice(&page.body[at..at + seg as usize]);
                at += seg as usize;
                if seg < 255 {
                    got.push(std::mem::take(&mut cur));
                }
            }
            assert_eq!(at, page.body.len(), "段表必须正好覆盖页体");
        }
        assert_eq!(r.crc_errors(), 0, "CRC 必须全对");
        assert_eq!(got, packets, "包内容与顺序必须原样回来");
        assert_eq!(last_granule, 6 * 1024, "末页 granule 是最后一个包的结束位置");
    }

    /// 一页里放几个包：`with_page_packets` 要真的生效。
    #[test]
    fn page_packet_limit_is_honoured() {
        let mut m = OggMuxer::new(1).with_page_packets(2);
        for _ in 0..5 {
            m.write_packet(&[1u8; 10], 0).unwrap();
        }
        let bytes = m.finish();
        let mut r = PageReader::new(&bytes[..]);
        let mut pages = 0;
        while let Some(p) = r.next_page().unwrap() {
            assert!(p.packet_count() <= 2, "一页不该超过 2 个包");
            pages += 1;
        }
        // 5 个包 / 每页 2 个 = 3 页（最后一页 1 个 + EOS）。
        assert_eq!(pages, 3);
    }

    /// BOS / EOS 只标在该标的地方。
    #[test]
    fn bos_and_eos_flags() {
        let mut m = OggMuxer::new(7).with_page_packets(1);
        m.write_packet(b"a", 0).unwrap();
        m.write_packet(b"b", 1).unwrap();
        let bytes = m.finish();
        let mut r = PageReader::new(&bytes[..]);
        let mut flags = Vec::new();
        while let Some(p) = r.next_page().unwrap() {
            flags.push((p.flags.bos(), p.flags.eos(), p.flags.continued()));
        }
        assert_eq!(
            flags,
            vec![(true, false, false), (false, true, false)],
            "第一页 BOS、最后一页 EOS，中间都不是"
        );
    }

    /// 显式 `flush()` 之后 `finish()` 要把 `EOS` **补到最后一页**上，
    /// 而不是另写一个空页。
    #[test]
    fn finish_patches_eos_onto_the_last_flushed_page() {
        let mut m = OggMuxer::new(3);
        m.write_packet(b"a", 0).unwrap();
        m.flush();
        m.write_packet(b"b", 1).unwrap();
        m.flush();
        let bytes = m.finish();

        let mut r = PageReader::new(&bytes[..]);
        r.set_verify_crc(true);
        let mut flags = Vec::new();
        while let Some(p) = r.next_page().unwrap() {
            flags.push((p.flags.bos(), p.flags.eos()));
        }
        assert_eq!(
            flags,
            vec![(true, false), (false, true)],
            "EOS 应该补在最后一页上，且不能多出空页"
        );
        // ⚠️ 补 EOS 改了一个字节，那一页的 CRC 必须重算 —— 忘了重算就是坏页，
        // 而播放器**不一定**报错（很多实现默认不校验）。
        assert_eq!(r.crc_errors(), 0, "补 EOS 之后 CRC 必须重算对");
    }
}
