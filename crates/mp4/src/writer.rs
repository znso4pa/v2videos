//! ISOBMFF 的字节原语：**大端**写入 + box 拼装。
//!
//! 每个 box 的布局都是 `size(4) + type(4) + body`，而 `size` 要等 body 拼完才知道。
//! 所以先往 `body` 里攒，`finish()` 时再补头 —— 省掉「先算长度再写一遍」的重复。
//!
//! ⚠️ ISOBMFF 全是**大端**。ASF / AVI 全是小端，切换时最容易写错的就是这里。

/// 一个 box 的构造器。
#[derive(Debug)]
pub struct BoxBuilder {
    typ: [u8; 4],
    body: Vec<u8>,
}

impl BoxBuilder {
    pub fn new(typ: &[u8; 4]) -> Self {
        Self {
            typ: *typ,
            body: Vec::new(),
        }
    }

    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.body.push(v);
        self
    }

    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.body.extend_from_slice(&v.to_be_bytes());
        self
    }

    pub fn i16(&mut self, v: i16) -> &mut Self {
        self.body.extend_from_slice(&v.to_be_bytes());
        self
    }

    /// 24 位（`flags` 字段）。
    pub fn u24(&mut self, v: u32) -> &mut Self {
        self.body.extend_from_slice(&v.to_be_bytes()[1..]);
        self
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.body.extend_from_slice(&v.to_be_bytes());
        self
    }

    /// `version`(8) + `flags`(24) —— 所有 full box 的开头。
    pub fn full(&mut self, version: u8, flags: u32) -> &mut Self {
        self.u8(version);
        self.u24(flags)
    }

    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.body.extend_from_slice(b);
        self
    }

    pub fn zeros(&mut self, n: usize) -> &mut Self {
        self.body.resize(self.body.len() + n, 0);
        self
    }

    /// 单位矩阵（3×3，16.16 定点）。ISOBMFF 的 `matrix` 字段。
    pub fn identity_matrix(&mut self) -> &mut Self {
        for v in [
            0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000,
        ] {
            self.u32(v);
        }
        self
    }

    /// 嵌一个子 box（先 `finish()` 成字节再接上）。
    pub fn child(&mut self, child: BoxBuilder) -> &mut Self {
        self.body.extend_from_slice(&child.finish());
        self
    }

    /// 接一段**已经拼好**的 box 字节（比如 `stsd` 的 sample entry）。
    pub fn raw(&mut self, box_bytes: &[u8]) -> &mut Self {
        self.body.extend_from_slice(box_bytes);
        self
    }

    /// 补上 `size + type`，产出完整 box。
    pub fn finish(self) -> Vec<u8> {
        let size = 8 + self.body.len();
        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(&(size as u32).to_be_bytes());
        out.extend_from_slice(&self.typ);
        out.extend_from_slice(&self.body);
        out
    }
}

/// MPEG-4 的「描述符长度」编码：每字节 7 位，最高位表示**还有后续字节**。
///
/// 用**最少字节数**（ffmpeg 一律写 4 字节的展开形式，两种都合法；我们取最省的那种）。
pub fn put_descr_len(out: &mut Vec<u8>, mut len: usize) {
    let mut buf = [0u8; 4];
    let mut n = 0;
    loop {
        buf[n] = (len & 0x7f) as u8;
        len >>= 7;
        n += 1;
        if len == 0 {
            break;
        }
    }
    // 大端序输出：`buf[n-1]` 是最高 7 位（第一个字节），`buf[0]` 是最低（最后一个）。
    // 除最后一个字节外都要置「续接」位。
    for i in (0..n).rev() {
        out.push(buf[i] | if i == 0 { 0 } else { 0x80 });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_size_covers_header_and_body() {
        let mut b = BoxBuilder::new(b"test");
        b.u32(0x1234_5678);
        let out = b.finish();
        assert_eq!(&out[..4], &[0, 0, 0, 12]);
        assert_eq!(&out[4..8], b"test");
        assert_eq!(&out[8..], &[0x12, 0x34, 0x56, 0x78]);
    }

    #[test]
    fn full_box_writes_version_then_flags() {
        let mut b = BoxBuilder::new(b"abcd");
        b.full(1, 0x00_00_07);
        let out = b.finish();
        assert_eq!(&out[8..], &[1, 0, 0, 7]);
    }

    #[test]
    fn descriptor_length_uses_minimal_bytes() {
        let mut out = Vec::new();
        put_descr_len(&mut out, 0x4e);
        assert_eq!(out, vec![0x4e]);

        out.clear();
        put_descr_len(&mut out, 0x80);
        assert_eq!(out, vec![0x81, 0x00]);

        out.clear();
        put_descr_len(&mut out, 0x3fff);
        assert_eq!(out, vec![0xff, 0x7f]);

        out.clear();
        put_descr_len(&mut out, 0x4000);
        assert_eq!(out, vec![0x81, 0x80, 0x00]);
    }

    #[test]
    fn identity_matrix_is_the_isobmff_one() {
        let mut b = BoxBuilder::new(b"mtrx");
        b.identity_matrix();
        let out = b.finish();
        let mut want = Vec::new();
        for v in [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
            want.extend_from_slice(&v.to_be_bytes());
        }
        assert_eq!(&out[8..], &want[..]);
    }
}
