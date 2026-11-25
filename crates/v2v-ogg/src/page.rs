//! Ogg 页的字节层：切页、校验、拆包。
//!
//! 一个 Ogg 文件就是**一串页**，没有索引、没有文件头 —— 页自己就是全部的元数据。
//!
//! ```text
//!  0        4  5        14        18      22      26   27      27+n
//!  +--------+--+--------+--------+-------+-------+----+-------+--------+
//!  | "OggS" |v | type   |granule |serial |  seq  |crc | nseg  | lacing | body…
//!  +--------+--+--------+--------+-------+-------+----+-------+--------+
//!   4        1    1        8        4        4      4     1     nseg
//! ```
//!
//! ## 两个容易搞错的地方
//!
//! 1. **`granule_position` 是 `i64`，`-1` 表示「本页没有任何包结束」**，
//!    不是一个巨大的正数。当 `u64` 读会让 `-1` 变成 `0xFFFF_FFFF_FFFF_FFFF`，
//!    下游一算时间戳就溢出。
//! 2. **段表（lacing）里 `255` 表示「这个包还没完，接着下一个段」**，
//!    只有 `< 255` 的段才**结束**一个包。所以一个包可以横跨任意多个段、
//!    进而横跨多页（跨页时下一页的 `continued` 位置 1）。
//!    把「段」当「包」会让每个大包被切成好几块 —— 而且解出来的音频
//!    听起来只是「有杂音」，不会报错。

use std::io::Read;

use crate::crc;
use crate::error::{Error, Result};

/// 页标识。
pub const CAPTURE: &[u8; 4] = b"OggS";

/// 固定页头的字节数（不含段表）。
pub const HEADER_SIZE: usize = 27;

/// 页体的最大字节数（255 个段 × 每段最多 255 字节）。
pub const MAX_BODY: usize = 255 * 255;

/// `granule_position` 里表示「没有包结束」的哨兵值。
pub const GRANULE_NONE: i64 = -1;

const FLAG_CONTINUED: u8 = 0x01;
const FLAG_BOS: u8 = 0x02;
const FLAG_EOS: u8 = 0x04;

/// 页头里的标志位，按语义拆开 —— 比让调用方自己 `& 0x04` 好读。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PageFlags(pub u8);

impl PageFlags {
    /// 本页的第一个包是上一页的续接。
    pub fn continued(self) -> bool {
        self.0 & FLAG_CONTINUED != 0
    }
    /// 本页是逻辑流的**第一页**（Beginning Of Stream）。
    pub fn bos(self) -> bool {
        self.0 & FLAG_BOS != 0
    }
    /// 本页是逻辑流的**最后一页**（End Of Stream）。
    pub fn eos(self) -> bool {
        self.0 & FLAG_EOS != 0
    }
}

/// 一页的元数据 + 载荷。
#[derive(Debug, Clone)]
pub struct Page {
    pub flags: PageFlags,
    /// 页内**最后一个完成的包**结束时的 PCM 位置（Vorbis 口径：采样数）。
    /// 本页没有包结束时是 [`GRANULE_NONE`]。
    pub granule: i64,
    /// 逻辑流编号。多流复用的文件靠它区分。
    pub serial: u32,
    /// 页序号，从 0 起，同一 `serial` 内连续。缺页可以靠它发现。
    pub sequence: u32,
    /// 文件里的原始 CRC（算过之后是否相符见 [`PageReader::crc_errors`]）。
    pub crc: u32,
    /// 段表：每项是这一段的字节数，`255` 表示包未结束。
    pub segments: Vec<u8>,
    /// 页体（所有段拼起来）。
    pub body: Vec<u8>,
}

impl Page {
    /// 本页里**结束**了几个包（段表里 `< 255` 的项数）。
    pub fn packet_count(&self) -> usize {
        self.segments.iter().filter(|&&s| s < 255).count()
    }

    /// 本页的字节长度（页头 + 段表 + 页体）。
    pub fn byte_len(&self) -> usize {
        HEADER_SIZE + self.segments.len() + self.body.len()
    }
}

/// 按页读 `R`。
///
/// 只做「字节 → 页」，不做拆包、不做时间戳 —— 那两层在 [`crate::PacketReader`]。
pub struct PageReader<R> {
    r: R,
    verify_crc: bool,
    crc_errors: u64,
    /// 已经读了多少字节，用来在报错时给出偏移。
    pos: u64,
    eof: bool,
}

impl<R: Read> PageReader<R> {
    pub fn new(r: R) -> Self {
        Self {
            r,
            // **默认不校验**：CRC 要对每页的每个字节查一次表，是纯粹的额外成本，
            // 而 ffmpeg 默认也不校验（它只在 `-err_detect crccheck` 时算）。
            // 真正的用途是**测试与排查**，所以做成可开。
            verify_crc: false,
            crc_errors: 0,
            pos: 0,
            eof: false,
        }
    }

    /// 打开整页 CRC 校验；不符时报 [`Error::Bad`]。
    pub fn set_verify_crc(&mut self, on: bool) {
        self.verify_crc = on;
    }

    /// 到目前为止 CRC 不符的页数（`set_verify_crc(false)` 时恒为 0）。
    pub fn crc_errors(&self) -> u64 {
        self.crc_errors
    }

    /// 已读字节数。
    pub fn position(&self) -> u64 {
        self.pos
    }

    /// 读下一页；文件正常结束返回 `Ok(None)`。
    ///
    /// 「正常结束」和「页中间断了」是两回事：前者是文件末尾正好在页边界上，
    /// 后者是数据被截断。截断会报错而不是悄悄少几页 —— 少的那几页里
    /// 可能有 EOS，也可能有音频，静默跳过会变成「音频莫名其妙短了一截」。
    pub fn next_page(&mut self) -> Result<Option<Page>> {
        if self.eof {
            return Ok(None);
        }

        let mut head = [0u8; HEADER_SIZE];
        match self.read_exact_or_eof(&mut head)? {
            false => {
                self.eof = true;
                return Ok(None);
            }
            true => {}
        }

        if &head[..4] != CAPTURE {
            return Err(Error::Bad(format!(
                "偏移 {} 处不是 Ogg 页（期望 \"OggS\"，实际 {:02X?}）",
                self.pos,
                &head[..4]
            )));
        }
        // 版本必须为 0。为别的值说明这不是我们认识的 Ogg，别硬猜。
        if head[4] != 0 {
            return Err(Error::Unsupported(format!(
                "Ogg 流版本 {}（只认 0）",
                head[4]
            )));
        }

        let flags = PageFlags(head[5]);
        let granule = i64::from_le_bytes(head[6..14].try_into().unwrap());
        let serial = u32::from_le_bytes(head[14..18].try_into().unwrap());
        let sequence = u32::from_le_bytes(head[18..22].try_into().unwrap());
        let stored_crc = u32::from_le_bytes(head[22..26].try_into().unwrap());
        let nseg = head[26] as usize;

        let mut segments = vec![0u8; nseg];
        if nseg > 0 {
            self.read_exact(&mut segments)?;
        }
        let body_len: usize = segments.iter().map(|&s| s as usize).sum();
        let mut body = vec![0u8; body_len];
        if body_len > 0 {
            self.read_exact(&mut body)?;
        }

        if self.verify_crc {
            // 校验范围是整页，crc 字段按 0 算 —— 分三段增量算，不复制整页。
            let mut c = crc::crc32_update(0, &head[..22]);
            c = crc::crc32_update(c, &[0, 0, 0, 0]);
            c = crc::crc32_update(c, &head[26..]);
            c = crc::crc32_update(c, &segments);
            c = crc::crc32_update(c, &body);
            if c != stored_crc {
                self.crc_errors += 1;
                return Err(Error::Bad(format!(
                    "页 CRC 不符（serial {serial:#010x} seq {sequence}）：存的 {stored_crc:#010x}，算出 {c:#010x}"
                )));
            }
        }

        Ok(Some(Page {
            flags,
            granule,
            serial,
            sequence,
            crc: stored_crc,
            segments,
            body,
        }))
    }

    /// 读满 `buf`；一个字节都读不到时返回 `false`，读了一半就断则报错。
    fn read_exact_or_eof(&mut self, buf: &mut [u8]) -> Result<bool> {
        let mut filled = 0;
        while filled < buf.len() {
            match self.r.read(&mut buf[filled..]) {
                Ok(0) => {
                    if filled == 0 {
                        return Ok(false);
                    }
                    return Err(Error::Bad(format!(
                        "文件在偏移 {} 处被截断（页头只读到 {filled}/{} 字节）",
                        self.pos,
                        buf.len()
                    )));
                }
                Ok(n) => {
                    filled += n;
                    self.pos += n as u64;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(true)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.read_exact_or_eof(buf)?
            .then_some(())
            .ok_or_else(|| Error::Bad(format!("文件在偏移 {} 处被截断", self.pos)))
    }
}

/// 把一个页体按段表切成若干个「本页内结束的包」。
///
/// 返回的每一项都是**完整**的包 —— 页尾那个没结束的包不在里面，
/// 它要靠 [`crate::PacketReader`] 跟下一页拼。
///
/// `prefix` 是上一页遗留的半截包（本页 `continued` 时用）。
pub(crate) fn split_packets<'a>(page: &'a Page, prefix: &mut Vec<u8>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut cur: Vec<u8> = std::mem::take(prefix);
    let mut off = 0usize;
    for &seg in &page.segments {
        let n = seg as usize;
        cur.extend_from_slice(&page.body[off..off + n]);
        off += n;
        if seg < 255 {
            out.push(std::mem::take(&mut cur));
        }
    }
    // 没结束的部分留给下一页。
    *prefix = cur;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一页合法字节（CRC 按规范算）。
    fn build_page(
        flags: u8,
        granule: i64,
        serial: u32,
        seq: u32,
        segments: &[u8],
        body: &[u8],
    ) -> Vec<u8> {
        assert_eq!(segments.iter().map(|&s| s as usize).sum::<usize>(), body.len());
        let mut p = Vec::new();
        p.extend_from_slice(CAPTURE);
        p.push(0);
        p.push(flags);
        p.extend_from_slice(&granule.to_le_bytes());
        p.extend_from_slice(&serial.to_le_bytes());
        p.extend_from_slice(&seq.to_le_bytes());
        p.extend_from_slice(&[0, 0, 0, 0]); // crc 占位
        p.push(segments.len() as u8);
        p.extend_from_slice(segments);
        p.extend_from_slice(body);
        let c = crc::crc32_page(&p, 22);
        p[22..26].copy_from_slice(&c.to_le_bytes());
        p
    }

    #[test]
    fn reads_a_single_page() {
        let bytes = build_page(FLAG_BOS, 0, 0x1234, 0, &[4], b"abcd");
        let mut r = PageReader::new(&bytes[..]);
        r.set_verify_crc(true);
        let p = r.next_page().unwrap().unwrap();
        assert!(p.flags.bos());
        assert!(!p.flags.continued());
        assert_eq!(p.granule, 0);
        assert_eq!(p.serial, 0x1234);
        assert_eq!(p.sequence, 0);
        assert_eq!(p.body, b"abcd");
        assert_eq!(p.packet_count(), 1);
        assert!(r.next_page().unwrap().is_none());
    }

    #[test]
    fn granule_minus_one_survives_as_negative() {
        // 这条是防「用 u64 读 granule」的：-1 必须读成 -1，不是 0xFFFF…
        let bytes = build_page(0, -1, 7, 1, &[1], b"x");
        let mut r = PageReader::new(&bytes[..]);
        let p = r.next_page().unwrap().unwrap();
        assert_eq!(p.granule, GRANULE_NONE);
        assert_eq!(p.granule, -1);
    }

    #[test]
    fn crc_mismatch_is_reported_only_when_enabled() {
        // 页布局：0..27 页头，27 是段表第一项，28 起才是页体。
        // 改的必须是**页体**，改成段表会先撞上「长度对不上」的截断检查。
        let mut bytes = build_page(0, 0, 1, 0, &[2], b"hi");
        assert_eq!(bytes[27], 2, "段表第一项");
        bytes[28] ^= 0xFF; // 改坏页体
        // 默认不校验 → 照读不误
        let mut r = PageReader::new(&bytes[..]);
        assert!(r.next_page().is_ok());
        // 开了校验 → 报错
        let mut r = PageReader::new(&bytes[..]);
        r.set_verify_crc(true);
        let e = r.next_page().unwrap_err();
        assert!(matches!(e, Error::Bad(ref s) if s.contains("CRC")), "{e}");
        assert_eq!(r.crc_errors(), 1);
    }

    #[test]
    fn truncated_page_is_an_error_not_eof() {
        let bytes = build_page(0, 0, 1, 0, &[8], b"12345678");
        for cut in [3usize, 27, 30, 34] {
            let mut r = PageReader::new(&bytes[..cut]);
            let e = r.next_page().unwrap_err();
            assert!(matches!(e, Error::Bad(ref s) if s.contains("截断")), "cut={cut}: {e}");
        }
    }

    #[test]
    fn bad_capture_pattern_is_rejected() {
        let mut bytes = build_page(0, 0, 1, 0, &[1], b"x");
        bytes[1] = b'X';
        let mut r = PageReader::new(&bytes[..]);
        assert!(matches!(r.next_page().unwrap_err(), Error::Bad(_)));
    }

    #[test]
    fn splits_packets_and_carries_the_unfinished_one() {
        // 页里：一个 3 字节的完整包 + 一个 255 字节的**未结束**包（段值是 255）。
        // 注意最后一个段必须是 255 才算「没完」——`[3, 255, 5]` 里那个 5 会把包结束掉。
        let mut body = Vec::new();
        body.extend_from_slice(b"abc");
        body.extend_from_slice(&[0xAAu8; 255]);
        let page = Page {
            flags: PageFlags(0),
            granule: 0,
            serial: 1,
            sequence: 0,
            crc: 0,
            segments: vec![3, 255],
            body,
        };
        let mut prefix = Vec::new();
        let pk = split_packets(&page, &mut prefix);
        assert_eq!(pk.len(), 1, "只有一个包在本页内结束");
        assert_eq!(pk[0], b"abc");
        assert_eq!(prefix.len(), 255, "没结束的 255 字节留给下一页");
    }

    #[test]
    fn continued_page_prepends_the_carried_bytes() {
        // 上一页留了 255 字节（那些字节在**上一页的页体**里，不在本页），
        // 本页 `continued`，段表 [2] 把它结束掉 → 这个包一共 257 字节。
        let page = Page {
            flags: PageFlags(FLAG_CONTINUED),
            granule: 10,
            serial: 1,
            sequence: 1,
            crc: 0,
            segments: vec![2],
            body: b"yz".to_vec(),
        };
        let mut prefix = vec![0xAAu8; 255];
        let pk = split_packets(&page, &mut prefix);
        assert_eq!(pk.len(), 1);
        assert_eq!(pk[0].len(), 257);
        assert_eq!(&pk[0][..255], &[0xAAu8; 255][..]);
        assert_eq!(&pk[0][255..], b"yz");
        assert!(prefix.is_empty(), "包已经结束，不该再有遗留");
    }
}
