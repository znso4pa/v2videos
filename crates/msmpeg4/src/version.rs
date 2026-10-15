//! MS-MPEG-4 家族的**五个版本** —— 五套语法共用一个骨架。
//!
//! | 版本 | 容器里的 FourCC | 俗称 | 与 V3（MP43）的关系 |
//! |---|---|---|---|
//! | `V1` | `MP41` | MS-MPEG-4 v1 | 帧头有 32 位起始码；MB 层用 **H.263** 的表 |
//! | `V2` | `MP42` | MS-MPEG-4 v2 | MB 层用 `v2_*` 两张私有表 |
//! | `V3` | `MP43` | MS-MPEG-4 v3 | **本仓最早做的那一个** |
//! | `Wmv1` | `WMV1` | WMV7 | **MB 层与 V3 完全相同**（ffmpeg 里共用 `msmpeg4v34_decode_mb`） |
//! | `Wmv2` | `WMV2` | WMV8 | 帧头与 MB 层都是自己一套（ffmpeg 里在 `wmv2dec.c`） |
//!
//! ⚠️ **`V3` 与 `Wmv1` 不是同一个东西**，但比名字看起来近得多：
//!
//! - 帧头：`Wmv1` 多读一个 **ext header**（fps 5 + 码率 11 + `flipflop_rounding` 1）、
//!   多一个 `per_mb_rl_table` 位，且 `inter_intra_pred` 由「尺寸 + 码率」推出来；
//! - 系数：`Wmv1` 的**第三 escape 是变长的**（`esc3_level_length`），V3 是定长 1+6+8；
//! - MB 层：**一模一样**。
//!
//! 这些不是「差不多」—— 差一个位整帧就解错了，所以每个版本都要单独对 ffmpeg 验证。

/// MS-MPEG-4 的版本。默认 `V3`（`.wmv` 里最常见的那一种）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Version {
    /// `MP41`。
    V1,
    /// `MP42`。
    V2,
    /// `MP43`。
    #[default]
    V3,
    /// `WMV1`（WMV7）。
    Wmv1,
    /// `WMV2`（WMV8）。
    Wmv2,
}

impl Version {
    /// 从容器里的 FourCC 认版本。认不出返回 `None`。
    pub fn from_fourcc(fourcc: u32) -> Option<Self> {
        Some(match &fourcc.to_le_bytes() {
            b"MP41" => Version::V1,
            b"MP42" => Version::V2,
            b"MP43" => Version::V3,
            b"WMV1" => Version::Wmv1,
            b"WMV2" => Version::Wmv2,
            _ => return None,
        })
    }

    /// FourCC 的显示名（报错信息用）。
    pub fn name(self) -> &'static str {
        match self {
            Version::V1 => "MP41",
            Version::V2 => "MP42",
            Version::V3 => "MP43",
            Version::Wmv1 => "WMV1",
            Version::Wmv2 => "WMV2",
        }
    }

    /// ffmpeg 的 `msmpeg4_version >= MSMP4_WMV1` —— 也就是「WMV1 与 WMV2」这一档。
    ///
    /// 它在**两处**起作用，而且两处都不是「细节」：
    /// 帧内块的 `run_diff`（1 而不是 0）、以及第三 escape 的定长/变长之分。
    pub fn is_wmv(self) -> bool {
        matches!(self, Version::Wmv1 | Version::Wmv2)
    }

    /// `msmpeg4_version <= MSMP4_V2` —— MP41 / MP42 这一档。
    pub fn is_v1_or_v2(self) -> bool {
        matches!(self, Version::V1 | Version::V2)
    }

    /// 帧内块的 `run_diff`：`version >= MSMP4_WMV1` → 1，否则 0。
    pub fn intra_run_diff(self) -> i32 {
        i32::from(self.is_wmv())
    }

    /// 帧间块的 `run_diff`：**只有 V2 是 0**，其余都是 1。
    ///
    /// （ffmpeg 写的是 `if (version == MSMP4_V2) run_diff = 0; else run_diff = 1;`
    /// —— 注意它是「只有 V2 特殊」，不是「只有 WMV 特殊」，两者不一样。）
    pub fn inter_run_diff(self) -> i32 {
        i32::from(self != Version::V2)
    }
}
