//! MPEG-4 Part 2 的头：**视觉对象序列（VOS）/ 视觉对象 / 视频对象层（VOL）/ 视频对象平面（VOP）**。
//!
//! ## 为什么 VOL 要单独抽出来
//!
//! VOL 是这条流里唯一**同时**出现在两个地方的头：
//!
//! 1. 编码后写进基本流（每个 I 帧前面重写一次，让流可以任意位置起播）；
//! 2. 原样塞进 MP4 的 `esds` → `DecoderSpecificInfo`，播放器**不解码任何一帧**
//!    就能从这 40 多个字节里读出宽高、帧率、profile。
//!
//! 两处必须**逐位相同**，所以这里只留一个 `write_vol`，两边都调它。
//!
//! ## 字段顺序是逐位核过 ffmpeg 的
//!
//! 顺序来自 `mpeg4videoenc.c` 的 `mpeg4_encode_visual_object_header` /
//! `mpeg4_encode_vol_header` / `ff_mpeg4_encode_picture_header`。
//! 这个顺序**不能靠猜**：MPEG-4 的头是纯位域、没有起始码以外的同步点，
//! 错一位后面全错，而且解码器往往不报错、只是画出一团糊。
//!
//! 校验办法：把生成的 VOL 拿去解析，字段应当与参考样本逐项相同 ——
//! 见 `tests/reference.rs` 里的 `vol_matches_the_reference_dsi`。
//!
//! ## `resync_marker_disable` 为什么写 0（而不是 1）
//!
//! 我们**不**往流里插 resync marker。这个标志位在规范里的含义是
//! 「VOP 里**不允许**出现 resync marker」（写 1 才是禁止），写 0 只是说「允许」，
//! 编码器仍可自行决定不插 —— 所以写 0 是合法的，且与 ffmpeg 的输出逐位一致，
//! 便于直接对账。ffmpeg 的解码器根本不看这一位。

use v2v_msmpeg4::BitWriter;

/// 视觉对象序列起始码（`VOS_STARTCODE`）。
pub const VOS_STARTCODE: u32 = 0x0000_01B0;
/// 视觉对象起始码（`VISUAL_OBJ_STARTCODE`）。
pub const VISUAL_OBJ_STARTCODE: u32 = 0x0000_01B5;
/// 用户数据起始码（`USER_DATA_STARTCODE`）。
pub const USER_DATA_STARTCODE: u32 = 0x0000_01B2;
/// 视频对象平面起始码（`VOP_STARTCODE`）。
pub const VOP_STARTCODE: u32 = 0x0000_01B6;

/// Simple Profile、Level 1 —— 320×240@25 落在这一档。
pub const PROFILE_SIMPLE_LEVEL1: u8 = 0x01;

/// `ff_mpeg4_stuffing`：先补一个 `0`，再用 `1` 填到字节边界。
///
/// 补 `0` 是规范要求的「stuffing 起始位」，之后的 `1` 是填充。两者都要写，
/// 少写 `0` 会让解码器把填充的 `1` 当成下一个语法元素。
fn stuffing(w: &mut BitWriter) {
    w.put(0, 1);
    let used = w.bit_pos() % 8;
    if used != 0 {
        let n = (8 - used) as u32;
        w.put((1u32 << n) - 1, n);
    }
}

/// `av_log2(x)`：`x` 的最高位位置（`x <= 1` 时为 0）。ffmpeg 的 `av_log2(0)` 也是 0。
fn log2(v: u32) -> u32 {
    if v <= 1 {
        0
    } else {
        31 - v.leading_zeros()
    }
}

/// VOP 头里 `vop_time_increment` 的位宽。
///
/// `av_log2(framerate.num - 1) + 1`（`mpegvideo_enc.c:687`）—— 25 fps → 5 位，
/// 所以 `time_mod` 以 25 为周期回绕，回绕时 VOP 头里要多写一个 `1` 位
/// （见 `VopParams::time_incr`）。**这个值必须和解码端算出来的一致**，
/// 否则 VOP 头之后的宏块数据会整体错位。
pub fn time_increment_bits(fps_num: u32) -> u32 {
    log2(fps_num.saturating_sub(1)) + 1
}

/// 视觉对象序列头 + 视觉对象头。整条流最前面写一次。
pub fn write_visual_object_header(w: &mut BitWriter, profile_and_level: u8) {
    w.put(VOS_STARTCODE, 32);
    w.put(u32::from(profile_and_level), 8);

    w.put(VISUAL_OBJ_STARTCODE, 32);
    w.put(1, 1); // visual_object_identifier = 1（后面跟着 ver_id / priority）
    let vo_ver_id = if profile_and_level >> 4 == 0xF { 5 } else { 1 };
    w.put(vo_ver_id, 4);
    w.put(1, 3); // visual_object_priority
    w.put(1, 4); // visual_object_type = 1（video）
    w.put(0, 1); // video_signal_type = 0
    stuffing(w);
}

/// 视频对象层（VOL）的参数。
#[derive(Debug, Clone, Copy)]
pub struct VolParams {
    pub width: u32,
    pub height: u32,
    /// `vop_time_increment_resolution`，取帧率的分子（25 fps → 25）。
    pub fps_num: u32,
    pub low_delay: bool,
    /// `quant_type`：`false` = H.263 量化（平坦步长），
    /// `true` = MPEG 量化（**逐系数乘规范默认矩阵**，见 `encode::Quant`）。
    ///
    /// ⚠️ 两者的**反量化规则完全不同**（帧间还多一个失配控制），编解码两侧必须一致 ——
    /// 只改这一位而不同时改量化器，解出来是一片雪花。
    pub mpeg_quant: bool,
}

/// 视频对象层头（`00 00 01 00` + `00 00 01 20` + 位域）。
///
/// 这份输出**同时**是 `esds` 里的 DSI，两处必须逐位相同。
pub fn write_vol(w: &mut BitWriter, p: &VolParams) {
    assert!(p.width < (1 << 13), "VOL 的宽度是 13 位");
    assert!(p.height < (1 << 13), "VOL 的高度是 13 位");
    assert!(
        p.fps_num < (1 << 16),
        "vop_time_increment_resolution 是 16 位"
    );
    assert!(p.fps_num > 0, "帧率分子不能是 0");

    w.put(0x100, 32); // 视频对象起始码（vo_number = 0）
    w.put(0x120, 32); // 视频对象层起始码（vol_number = 0）

    w.put(0, 1); // random_accessible_vol = 0
    w.put(1, 8); // video_object_type_indication = 1（Simple）
    w.put(1, 1); // is_object_layer_identifier = 1
    w.put(1, 4); // video_object_layer_verid = 1
    w.put(1, 3); // video_object_layer_priority = 1
    w.put(1, 4); // aspect_ratio_info = 1（方形像素，1:1）

    w.put(1, 1); // vol_control_parameters = 1
    w.put(1, 2); // chroma_format = 1（4:2:0）
    w.put(u32::from(p.low_delay), 1);
    w.put(0, 1); // vbv_parameters = 0

    w.put(0, 2); // video_object_layer_shape = 0（矩形）
    w.put(1, 1); // marker

    w.put(p.fps_num, 16); // vop_time_increment_resolution
    w.put(1, 1); // marker
    w.put(0, 1); // fixed_vop_rate = 0（不写 vop_time_increment 的定值表）
    w.put(1, 1); // marker

    w.put(p.width, 13);
    w.put(1, 1); // marker
    w.put(p.height, 13);
    w.put(1, 1); // marker

    w.put(0, 1); // interlaced = 0
    w.put(1, 1); // obmc_disable = 1
    w.put(0, 1); // sprite_enable = 0（verid == 1 时只有 1 位）
    w.put(0, 1); // not_8_bit = 0
    w.put(u32::from(p.mpeg_quant), 1); // quant_type
    if p.mpeg_quant {
        // 两张矩阵都**不传** —— 用规范默认的那对（`DEFAULT_INTRA_MATRIX`
        // / `DEFAULT_INTER_MATRIX`）。ffmpeg 的编码器也只置 `quant_type`：
        // 少 ~130 字节的载荷，产物与它对齐。
        w.put(0, 1); // load_intra_quant_mat
        w.put(0, 1); // load_non_intra_quant_mat
    }

    w.put(1, 1); // complexity_estimation_disable = 1
    w.put(0, 1); // resync_marker_disable = 0（见模块文档：写 0 但仍不插 marker）
    w.put(0, 1); // data_partitioned = 0
    w.put(0, 1); // scalable = 0

    stuffing(w);
}

/// 用户数据（`00 00 01 B2` + ASCII）。写自己的标识，不冒充别的编码器。
pub fn write_user_data(w: &mut BitWriter, s: &str) {
    w.put(USER_DATA_STARTCODE, 32);
    for b in s.bytes() {
        w.put(u32::from(b), 8);
    }
}

/// 帧类型 —— 只做 I 与 P。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VopKind {
    I,
    P,
}

/// VOP 头要写的东西。
#[derive(Debug, Clone, Copy)]
pub struct VopParams {
    pub kind: VopKind,
    /// 本帧在时间基里的余数（`frame_index % fps_num`）。
    pub time_mod: u32,
    /// 时间基回绕了多少次（`frame_index / fps_num` 的增量）。每个 `1` 位算一次。
    pub time_incr: u32,
    pub time_increment_bits: u32,
    pub qscale: u8,
    /// **只对 P 帧有效**：半像素舍入方式。0 = 四舍五入（`+r1/r2`），1 = 截断。
    /// 必须与运动补偿用的口径一致，否则编码器的重建会和解码器差半个像素。
    pub no_rounding: bool,
    /// **只对 P 帧有效**：运动矢量的 `f_code`（1..=7）。
    ///
    /// 它决定两件事：MV 差分的可表示范围（`±(2^(4+f_code) - 1)`）与**每个分量多写
    /// `f_code - 1` 个低位**。取 1 时范围是 `±31`（半像素单位），且不写低位 ——
    /// 我们的运动估计把差分夹在 `±31` 内，所以恒取 1。
    pub f_code: u8,
}

/// 一个 **I** 帧的 VOP 头（不含宏块数据）。
pub fn write_vop_header(w: &mut BitWriter, p: &VopParams) {
    assert!((1..=31).contains(&p.qscale), "qscale 是 5 位，1..=31");
    assert!(
        p.time_mod < (1 << p.time_increment_bits),
        "time_mod 超出 time_increment_bits 能表示的范围"
    );
    if p.kind == VopKind::P {
        assert!((1..=7).contains(&p.f_code), "f_code 是 3 位，1..=7");
    }

    w.put(VOP_STARTCODE, 32);
    // `pict_type - 1`：I = 0，P = 1。
    w.put(u32::from(p.kind == VopKind::P), 2);

    // 时间基回绕：每回绕一次补一个 1 位。
    for _ in 0..p.time_incr {
        w.put(1, 1);
    }
    w.put(0, 1); // 回绕串的结束位

    w.put(1, 1); // marker
    w.put(p.time_mod, p.time_increment_bits);
    w.put(1, 1); // marker
    w.put(1, 1); // vop_coded = 1

    // 只有 P 帧有这一位。
    if p.kind == VopKind::P {
        w.put(u32::from(p.no_rounding), 1);
    }

    // intra_dc_vlc_threshold：3 位**索引**，不是阈值本身。
    // 解码端 `intra_dc_threshold = ff_mpeg4_dc_threshold[bits]`，索引 0 → 99；
    // 而 `use_intra_dc_vlc = qscale < intra_dc_threshold`，qscale 最大 31 < 99，
    // 所以**帧内块永远走 DC 专用的 VLC**（`ff_mpeg4_DCtab_*`），不会掉到 AC 通道。
    // ffmpeg 的编码器也是恒写 0。
    w.put(0, 3);

    w.put(u32::from(p.qscale), 5);

    // 只有 P 帧有 `f_code`。
    if p.kind == VopKind::P {
        w.put(u32::from(p.f_code), 3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(f: impl FnOnce(&mut BitWriter)) -> Vec<u8> {
        let mut w = BitWriter::new();
        f(&mut w);
        w.finish()
    }

    #[test]
    fn stuffing_pads_to_a_byte_boundary_with_ones() {
        // 写 5 位，stuffing 应当补 `0` + `11` → 凑满一字节 `xxxx x011`。
        let got = bits(|w| {
            w.put(0b10101, 5);
            stuffing(w);
        });
        assert_eq!(got, vec![0b1010_1011]);
    }

    #[test]
    fn stuffing_after_a_full_byte_writes_a_whole_byte_of_padding() {
        // 已经字节对齐时，`ff_mpeg4_stuffing` 写 `0` + 补满 7 个 `1` → 整个字节 `0x7F`。
        // 那一个 `0` 位不能省（规范里的 stuffing 起始位），后面的 `1` 也不能省。
        let got = bits(|w| {
            w.put(0xAB, 8);
            stuffing(w);
        });
        assert_eq!(got, vec![0xAB, 0x7F]);
    }

    #[test]
    fn time_increment_bits_matches_ffmpeg() {
        // `av_log2(fps_num - 1) + 1`
        assert_eq!(time_increment_bits(1), 1);
        assert_eq!(time_increment_bits(2), 1);
        assert_eq!(time_increment_bits(24), 5); // av_log2(23) + 1 = 4 + 1
        assert_eq!(time_increment_bits(25), 5);
        assert_eq!(time_increment_bits(30), 5);
        assert_eq!(time_increment_bits(50), 6);
    }

    #[test]
    fn visual_object_header_is_the_reference_bytes() {
        // 参考样本 DSI 的开头 10 字节：`000001b0 01 000001b5 89 13`。
        // 最后一个 `13` = 1000 1001（identifier+verid+priority） + 0001（type=video）
        // + 0（video_signal_type）+ 0111（stuffing）。
        let got = bits(|w| write_visual_object_header(w, PROFILE_SIMPLE_LEVEL1));
        assert_eq!(
            got,
            vec![0x00, 0x00, 0x01, 0xB0, 0x01, 0x00, 0x00, 0x01, 0xB5, 0x89, 0x13]
        );
    }
}
