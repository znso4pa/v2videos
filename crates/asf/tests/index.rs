//! Simple Index 的**独立 oracle**：让 `ffmpeg` 自己去读我们写的索引。
//!
//! 按 AGENTS.md：自己写、自己读的往返测试**必要但不充分** —— 它和它编码的假设会一起错。
//! 索引的价值全在「别的播放器能不能用」，所以判据是**别家解析器读出来的是什么**。
//!
//! `ffmpeg` 的 asf 解封装器在 seek 时会解析 Simple Index 并打调试日志：
//!
//! ```text
//! [in#0/asf @ …] itime:0x989680, pct:2, ict:213
//! [in#0/asf @ …] pktnum:69, pktct:1  pts: 3900
//! ```
//!
//! 前三项就是索引头（时间间隔 / 最大包数 / 条目数），后面每行是一个**位置变了的**条目。
//! 这里断言 ffmpeg 读出来的三项与我们从文件里解析出的索引一致。
//!
//! 不需要真样本：索引是**容器层**的东西，用合成帧就能验；缺 `ffmpeg` 时打 `SKIP` 并通过。

use std::process::Command;

use v2v_asf::{AsfMuxer, VideoStreamInfo};

/// ffmpeg 用的索引时间间隔（100 ns 单位，1 秒）。`0x989680` == 10_000_000。
const ITIME_HEX: &str = "0x989680";

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// 从文件里解析出 `(条目数, 最大包数, 不重复的包号列表)`。
fn parse_index(file: &[u8]) -> (u32, u32, Vec<u32>) {
    const SIMPLE_INDEX: [u8; 16] = [
        0x90, 0x08, 0x00, 0x33, 0xB1, 0xE5, 0xCF, 0x11, 0x89, 0xF4, 0x00, 0xA0, 0xC9, 0x03, 0x49,
        0xCB,
    ];
    let at = file
        .windows(16)
        .position(|w| w == SIMPLE_INDEX)
        .expect("文件里应有 Simple Index 对象");
    let p = at + 16 + 8 + 16; // 跳过 长度字段 + file id
    let max_packet = u32::from_le_bytes(file[p + 8..p + 12].try_into().unwrap());
    let count = u32::from_le_bytes(file[p + 12..p + 16].try_into().unwrap());
    let mut packets = Vec::new();
    for i in 0..count as usize {
        let e = p + 16 + 6 * i;
        let n = u32::from_le_bytes(file[e..e + 4].try_into().unwrap());
        if packets.last() != Some(&n) {
            packets.push(n);
        }
    }
    (count, max_packet, packets)
}

/// 合成一段带 8 个关键帧的流，交给 ffmpeg 读索引。
#[test]
fn ffmpeg_reads_the_simple_index_we_wrote() {
    if !have_ffmpeg() {
        eprintln!("SKIP ffmpeg 不可用");
        return;
    }

    let mut m = AsfMuxer::new(VideoStreamInfo {
        width: 64,
        height: 48,
        fourcc: u32::from_le_bytes(*b"MP43"),
    });
    // 每 1 秒一个关键帧（第一个帧大一点，跨包），中间夹 4 个非关键帧。
    let mut pts = 0u32;
    for sec in 0..8u32 {
        let size = if sec == 2 { 9000 } else { 80 };
        let data: Vec<u8> = (0..size).map(|i| (i * 31 + sec as usize) as u8).collect();
        m.write_video(&data, true, pts).unwrap();
        for k in 1..5u32 {
            m.write_video(&[k as u8; 40], false, pts + k * 40).unwrap();
        }
        pts += 1000;
    }
    let file = m.finish();

    let (count, max_packet, packets) = parse_index(&file);
    assert!(count > 8, "preroll 段也要占槽，条目数应多于关键帧数");
    assert_eq!(max_packet, 3, "第 2 秒那个大帧占 3 包");

    let path = std::env::temp_dir().join(format!("v2v_asf_index_{}.wmv", std::process::id()));
    std::fs::write(&path, &file).unwrap();

    // `-ss` 触发 `asf_read_seek` → `asf_build_simple_index` → 打日志。
    // 帧内容是合成的，解码必然报错；我们只要 seek 那一段的日志，所以不看退出码。
    let out = Command::new("ffmpeg")
        .args(["-v", "debug", "-ss", "4", "-i"])
        .arg(&path)
        .args(["-f", "null", "-"])
        .output()
        .expect("run ffmpeg");
    let log = String::from_utf8_lossy(&out.stderr);
    let _ = std::fs::remove_file(&path);

    let header = log
        .lines()
        .find(|l| l.contains("itime:"))
        .unwrap_or_else(|| panic!("ffmpeg 没打索引日志 —— 它没读到索引？\n{log}"));
    assert!(
        header.contains(&format!("itime:{ITIME_HEX}")),
        "时间间隔对不上：{header}"
    );
    assert!(
        header.contains(&format!("pct:{max_packet}")),
        "最大包数对不上（我们写的是 {max_packet}）：{header}"
    );
    assert!(
        header.contains(&format!("ict:{count}")),
        "条目数对不上（我们写的是 {count}）：{header}"
    );

    // 条目：ffmpeg 只打「位置变了」的那些（同 pos 会被它去重），所以是**子集**。
    let seen: Vec<u32> = log
        .lines()
        .filter_map(|l| l.split("pktnum:").nth(1))
        .filter_map(|r| r.split(',').next())
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    assert!(!seen.is_empty(), "ffmpeg 一个条目都没打：\n{log}");
    for n in &seen {
        assert!(
            packets.contains(n),
            "ffmpeg 报的包号 {n} 不在我们写的槽里 {packets:?}"
        );
    }
    // 关键帧都该被 ffmpeg 看到（我们的槽按关键帧建，位置各不相同）。
    assert_eq!(
        seen.len(),
        packets.len(),
        "ffmpeg 看到的关键帧位置数与我们的槽数不一致\n我们: {packets:?}\nffmpeg: {seen:?}"
    );
}
