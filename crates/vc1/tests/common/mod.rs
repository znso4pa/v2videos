//! 对账用的夹具：真样本 + `ffmpeg` 的日志。
//!
//! ⚠️ 每个测试二进制都会把这个模块整个编进来，而它们各用一部分 ——
//! 所以这里允许「有些函数在当前这个二进制里没被用到」。
//!
//! 与 `crates/msmpeg4/tests/common/mod.rs` 同风格，但**只留这一层要用的**：
//! 这一层的 oracle 不是像素，而是 `ffmpeg -debug pict` 打出来的**逐帧帧头日志**。
//! （帧头的判据必须是外部 oracle：自己解、自己比等于把「位游标错位」和
//! 「对字段的理解错」两个错一起放过。）

#![allow(dead_code)]

use std::fs::File;
use std::io::BufReader;
use std::process::Command;

use v2v_asf::AsfDemuxer;
use v2v_container::StreamKind;

/// 样本路径列表；没设 / 文件不存在 → 空（调用方打 `SKIP`）。
///
/// 与别的 crate 共用同一个环境变量 —— 一份样本列表喂全套对账。
pub fn sample_list() -> Vec<String> {
    let Ok(v) = std::env::var("V2V_ASF_SAMPLE") else {
        return Vec::new();
    };
    v.split(':')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && std::path::Path::new(s).is_file())
        .map(|s| s.to_string())
        .collect()
}

/// `ffmpeg` / `ffprobe` 在不在 `PATH` 上。
///
/// ⚠️ **必须**：CI 上没装 ffmpeg，少了它这条测试会 panic，
/// 而仓规是「缺外部依赖就打个 `SKIP` 并通过」。
pub fn has_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// 视频轨：`(流号, 编码宽, 编码高, fourcc, extradata)`。
pub struct VideoTrack {
    pub number: u8,
    pub encoded_width: u32,
    pub encoded_height: u32,
    pub fourcc: u32,
    pub extra: Vec<u8>,
}

/// 取视频轨的元信息（拿不到就 `None`）。
pub fn video_track(path: &str) -> Option<VideoTrack> {
    let d = AsfDemuxer::new(BufReader::new(File::open(path).ok()?)).ok()?;
    let v = d.streams().iter().find(|s| s.kind == StreamKind::Video)?;
    let vf = v.video.as_ref()?;
    Some(VideoTrack {
        number: v.number,
        encoded_width: vf.encoded_width,
        encoded_height: vf.encoded_height,
        fourcc: vf.fourcc,
        extra: vf.extra.clone(),
    })
}

/// 按**解码顺序**取出全部视频帧（不过滤关键帧）。
pub fn all_video_frames(path: &str, vnum: u8) -> Option<Vec<Vec<u8>>> {
    let mut d = AsfDemuxer::new(BufReader::new(File::open(path).ok()?)).ok()?;
    let mut out = Vec::new();
    while let Some(s) = d.next_sample().ok()? {
        if s.stream == vnum {
            out.push(s.data);
        }
    }
    (!out.is_empty()).then_some(out)
}

/// `ffmpeg` 认的 fourcc（`WMV3`）。
pub const FOURCC_WMV3: u32 = u32::from_le_bytes(*b"WMV3");

/// `ffmpeg -debug pict` 日志里的一帧。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PictLog {
    /// `I` / `P` / `B` / `?`。
    pub pict_type: char,
    /// 「MV 类型」位平面：`(imode, invert)`。整帧 1MV 时 ffmpeg 不打印这行 → `None`。
    pub mv_type_plane: Option<(u8, u8)>,
    /// 「跳过」位平面：`(imode, invert)`。
    pub skip_plane: Option<(u8, u8)>,
}

/// 跑一遍 `ffmpeg -debug pict`，把**序列头里的 profile** 与逐帧日志一起抓回来。
///
/// ⚠️ 四处坑，全踩过：
/// - **`-loglevel repeat+debug` 里的 `repeat` 是必须的**：这些行是
///   `av_log(AV_LOG_DEBUG)`，默认的 `info` 级不会打（要 `debug`）；而 ffmpeg CLI
///   默认开着 `AV_LOG_SKIP_REPEATED`，**连续重复的行会被折叠成
///   「Last message repeated N times」** —— 连续几十个 `pict_type: P` 一模一样，
///   于是日志成了**子集**（实测 3719 帧只剩 3525 行）。子集不会报错，
///   只会让 `zip` 整体错位，看着像「第 10 帧起 pict_type 全错」。
///   `repeat` 前缀就是关掉这个折叠的。
/// - **`-debug pict`**（`FF_DEBUG_PICT_INFO`）是那些行外面那圈 `if`。
/// - **行序是「先位平面、后 `pict_type`」**：`pict_type` 那行在 `vc1_decode_frame()` 里、
///   **帧头解完之后**才打 —— 所以要先攒位平面行，等 `pict_type` 到了再落成一帧。
///   反了会整体错开一帧，而且因为相邻帧常常同类型，看起来「差不多对」。
/// - **探测期那个解码器也在打日志**：读入阶段 ffmpeg 会先开一个探测用的解码器试解
///   一两帧（另一个 `[wmv3 @ 0x…]` 上下文）。它的 `pict_type` 行混进来，帧数就比
///   `ffprobe -count_frames` 多一 —— 看着像「我们少解了一帧」，其实是日志脏了。
///   按上下文地址分组、取 `pict_type` 行最多的那一组，是这里最稳的挑法。
pub fn ffmpeg_pict_log(path: &str) -> Option<(u8, Vec<PictLog>)> {
    let out = Command::new("ffmpeg")
        .args(["-loglevel", "repeat+debug", "-debug", "pict", "-i", path])
        .args(["-f", "null", "-"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stderr);

    // 按 `[wmv3 @ 0x…]` 的地址分组；没有地址前缀的行（解码器之外的日志）丢掉。
    let ctx = |line: &str| -> Option<String> {
        let at = line.find(" @ 0x")?;
        let addr: String = line[at + 4..]
            .chars()
            .take_while(|c| c.is_ascii_hexdigit() || *c == 'x')
            .collect();
        Some(addr)
    };
    let mut groups: Vec<(String, Vec<&str>)> = Vec::new();
    for line in text.lines() {
        if let Some(a) = ctx(line) {
            match groups.iter_mut().find(|(k, _)| *k == a) {
                Some((_, v)) => v.push(line),
                None => groups.push((a, vec![line])),
            }
        }
    }
    // 真正那个解码器 = 打过最多 `pict_type` 的那组（探测用的只解一两帧）。
    let real = groups
        .iter()
        .max_by_key(|(_, v)| v.iter().filter(|l| l.contains("pict_type: ")).count())?;
    if real.1.iter().all(|l| !l.contains("pict_type: ")) {
        return None;
    }

    // `[wmv3 @ …] Profile 1:` —— 解码器自己解出来的 profile。
    let profile = real
        .1
        .iter()
        .find_map(|l| l.split("Profile ").nth(1))
        .and_then(|r| r.trim_end_matches(':').trim().parse::<u8>().ok())?;

    let mut frames: Vec<PictLog> = Vec::new();
    let mut mv_type: Option<(u8, u8)> = None;
    let mut skip: Option<(u8, u8)> = None;
    for line in &real.1 {
        if let Some(rest) = line.split("MB MV Type plane encoding:").nth(1) {
            mv_type = parse_imode(rest);
        } else if let Some(rest) = line.split("MB Skip plane encoding:").nth(1) {
            skip = parse_imode(rest);
        } else if let Some(rest) = line.split("pict_type: ").nth(1) {
            let pict_type = rest.trim().chars().next().unwrap_or('?');
            frames.push(PictLog {
                pict_type,
                mv_type_plane: mv_type.take(),
                skip_plane: skip.take(),
            });
        }
    }
    (!frames.is_empty()).then_some((profile, frames))
}

/// `Imode: 4, Invert: 0` -> `(4, 0)`。
fn parse_imode(s: &str) -> Option<(u8, u8)> {
    let imode = s.split("Imode:").nth(1)?.split(',').next()?.trim().parse().ok()?;
    let invert = s.split("Invert:").nth(1)?.trim().parse().ok()?;
    Some((imode, invert))
}
