//! 两条写在头文件里、但一直没人验过的承诺。
//!
//! 1. **错误通道是每线程的**（`v2v_last_error` 底下是 `thread_local!`）；
//! 2. **多个句柄互不干扰**（两个线程各拿一个，读同一个文件也不串）。
//!
//! 第 1 条特别值得钉死：把 `thread_local!` 换成一把全局 `Mutex`，**现有测试
//! 会全部继续绿** —— 单线程里两者行为完全一样，只有两个线程同时出错时才分得开。
//! 而一旦串了，症状是「错误消息对不上现场」，比没有错误消息更难查。

use std::ffi::{CStr, CString};
use std::sync::{Arc, Barrier};

use v2v_capi::common::{v2v_last_error, v2v_last_error_kind, EOS, KIND_BAD, KIND_IO, OK};

fn msg() -> String {
    let p = v2v_last_error();
    assert!(!p.is_null(), "v2v_last_error 不该返回空指针");
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// 一个一定不存在的路径。用它当错误消息里的「指纹」—— 消息里带路径，
/// 于是「读到的是自己的错误」可以被逐字验证，不用猜措辞。
fn missing(tag: &str) -> CString {
    CString::new(format!(
        "/tmp/v2v_abi_threads_{tag}_{}_no_such_file.wmv",
        std::process::id()
    ))
    .unwrap()
}

/// 两个线程各自制造一个错误，**在屏障上对齐之后**才去读。
///
/// 为什么必须对齐：如果错误通道是全局的，就是「谁最后写谁赢」—— 读得早的
/// 那个线程可能碰巧读到自己的，测试就时绿时红。对齐之后两个线程都已经写完，
/// 全局通道必然让其中一个读到别人的消息。
#[test]
fn error_channel_is_per_thread() {
    let paths = [missing("a"), missing("b")];
    let texts: Vec<String> = paths.iter().map(|p| p.to_str().unwrap().to_string()).collect();
    let barrier = Arc::new(Barrier::new(paths.len()));

    let mut hs = Vec::new();
    for (i, p) in paths.into_iter().enumerate() {
        let b = barrier.clone();
        let other = texts[1 - i].clone();
        hs.push(std::thread::spawn(move || {
            let h = unsafe { v2v_capi::demux::v2v_demux_open(p.as_ptr()) };
            assert!(h.is_null(), "不存在的文件不该打开成功");
            let kind = v2v_last_error_kind();
            b.wait(); // 两个线程都写完了，再读
            (p, other, kind, msg())
        }));
    }

    for h in hs {
        let (p, other, kind, m) = h.join().expect("线程 panic");
        let own = p.to_str().unwrap();
        assert_eq!(kind, KIND_IO, "缺失文件该是 KIND_IO，消息：{m}");
        assert!(m.contains(own), "线程读到的不是自己的错误：\n  期望含 {own}\n  实际 {m}");
        // 反过来说：别人那条路径不许出现在我的消息里。
        assert!(
            !m.contains(&other),
            "错误通道串了：我的消息里出现了别人的路径\n  {m}"
        );
    }
}

/// 主线程的错误不许被子线程覆盖 —— 宿主最常见的用法是「主线程调 API，
/// 工作线程在跑解码」，如果子线程一出错就把主线程的消息盖掉，主线程拿着
/// 一个成功码去读错误，会读到完全无关的内容。
#[test]
fn main_thread_error_survives_workers() {
    // 主线程先留一个自己的错误。
    let mut info: v2v_capi::player::V2vInfo = unsafe { std::mem::zeroed() };
    info.size = std::mem::size_of::<v2v_capi::player::V2vInfo>() as u32;
    let rc = unsafe { v2v_capi::player::v2v_player_info(std::ptr::null_mut(), &mut info) };
    assert_eq!(rc, v2v_capi::common::ERR_ARG);
    assert_eq!(v2v_last_error_kind(), KIND_BAD);
    let mine = msg();

    // 子线程制造一批别的错误。
    std::thread::spawn(|| {
        for i in 0..20 {
            let p = missing(&format!("w{i}"));
            let h = unsafe { v2v_capi::demux::v2v_demux_open(p.as_ptr()) };
            assert!(h.is_null());
        }
    })
    .join()
    .unwrap();

    assert_eq!(v2v_last_error_kind(), KIND_BAD, "主线程的错误种类被覆盖了");
    assert_eq!(msg(), mine, "主线程的错误消息被覆盖了");
}

/// 两个线程各开一个播放器，读同一个文件 —— 帧序列必须一模一样。
///
/// 这一条测的不是「线程安全」（句柄是 `!Send`，本来就只许单线程用），而是
/// **「两个句柄之间没有共享的可变状态」**：如果解码器内部有全局缓存 / 静态表
/// 被写，两边会互相污染，症状是帧序错乱或画面花 —— 而单线程测试永远看不见。
#[test]
fn two_handles_in_two_threads_do_not_interfere() {
    let Some(path) = std::env::var("V2V_ASF_SAMPLE")
        .ok()
        .and_then(|s| s.split(':').find(|p| std::path::Path::new(p).is_file()).map(String::from))
    else {
        return;
    };

    let pts_of = |path: &str| -> Vec<i64> {
        let c = CString::new(path).unwrap();
        let pl = unsafe { v2v_capi::player::v2v_player_open(c.as_ptr()) };
        assert!(!pl.is_null(), "打开失败：{}", msg());
        let mut out = Vec::new();
        for _ in 0..24 {
            let mut f: v2v_capi::player::V2vFrame = unsafe { std::mem::zeroed() };
            f.size = std::mem::size_of::<v2v_capi::player::V2vFrame>() as u32;
            let rc = unsafe { v2v_capi::player::v2v_player_read_video(pl, &mut f) };
            if rc == EOS {
                break;
            }
            assert_eq!(rc, OK, "读帧失败：{}", msg());
            out.push(f.pts_ms);
        }
        unsafe { v2v_capi::player::v2v_player_close(pl) };
        out
    };

    // 先单线程跑一遍当基准，再两个线程同时跑，比对。
    let base = pts_of(&path);
    assert!(!base.is_empty(), "一帧都没读到");

    let barrier = Arc::new(Barrier::new(2));
    let mut hs = Vec::new();
    for _ in 0..2 {
        let b = barrier.clone();
        let p = path.clone();
        hs.push(std::thread::spawn(move || {
            b.wait(); // 尽量让两边的读交错发生
            pts_of(&p)
        }));
    }
    for h in hs {
        let got = h.join().expect("线程 panic");
        assert_eq!(got, base, "两个句柄同时读，帧序列不一致 —— 有共享的可变状态");
    }
}
