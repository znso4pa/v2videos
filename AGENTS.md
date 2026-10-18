# AGENTS.md

给在这个仓库里干活的 AI 编码代理看的说明。

**v2video = 用 Rust 研究「视频之间的编码格式转换」**（容器 + 视频编码 + 音频编码）。
链路：**解封装 → 解码 → 像素/采样域 → 编码 → 封装**。
面向人的说明见 [README](README.md)；`TODO.md` 是变更日志 + 路线图，**唯一真源**（本地、不入库）。

## 现状

**链路已经打通到 `.wmv → .mp4`**：10 个 crate、约 5.5 万行 Rust，**零第三方依赖**
（`Cargo.lock` 里只有本仓的包）。

```text
container  解封装共用的数据模型（Stream / Sample / AudioFormat / VideoFormat）—— 零依赖
asf        ASF（.wmv / .wma）解封装 **+ 封装**（多载荷打包 + Simple Index）
avi        AVI（RIFF）解封装
msmpeg4    MP43 / WMV1 / WMV2 解码（I / P）**+ MP43 编码**（I / P，含运动估计）；兼**共享数值内核**（见下）
wma        WMA v1 / v2 解码（**只有解码器**）
vc1        VC-1（WMV3）解码（I / P，含 ABT / ¼ 像素 MC / 环路滤波；**只有解码器**）
mp4v       MPEG-4 Part 2 编码（I / P + 码率控制）
aac        AAC-LC 编码（MDCT / 短窗 / TNS / Huffman / 码率控制）
mp4        MP4 / ISOBMFF 封装（mp4v + mp4a(AAC) 或 sowt PCM）
cli        `v2v convert`：ASF / AVI → `.wmv`（MP43 + 音频直通）或 `.mp4`（mp4v + AAC）
```

⚠️ **`v2v-vc1` 还没接进 `v2v-cli`** —— 它能解 WMV3 出像素，但 `v2v convert` 仍不认这个
fourcc。接线时要处理「解码顺序 ≠ 显示顺序」（VC-1 的 I/P 按解码序进、按 PTS 出）。

下面的**约定**与**架构不变量**是硬约束（从隔壁 UsefulUnpack 的实战里换来的，不是摆设）——
动手前先读完；**进度、判据、下一步**在 `TODO.md`（本地、不入库、唯一真源），面向人的状态表在 README。

## 约定（现在起生效）

- **许可证是 LGPL-2.1-or-later。** 原因：MP43 解码需要的表（`crates/msmpeg4/src/tables.rs`）
  是从 FFmpeg 的数据文件读出来的（同为 LGPL）—— 与其搞「单文件例外」这种容易说错的
  安排，不如整个仓库统一按 LGPL 走。**代价**：GPL 侧的编码器（x264/x265/…）链不了，
  编码只能走宽松许可的实现或硬件（Android MediaCodec）；引第三方 crate 也要挑宽松许可的
  （Apache-2.0 / MIT / BSD），**MPL 这类 copyleft 一律不引**（UsefulUnpack 就是这么拒绝
  `symphonia` 的）。LGPL 作为**库**被 MIT 的应用依赖是可以的；别把 LGPL 代码并进
  MIT 的库再改许可。
- **提交信息**：`type(scope): 中文描述`。Subject 一行；正文 3–6 条 `-` 单行要点、不折行，
  说「改了什么、为什么重要」，不复述 diff。**考古细节**（试过哪些候选、某个夹具怎么骗人、
  量出来的数字）写 `TODO.md`，不进提交。
- **`TODO.md` 是本地文件、不入库**（在 `.gitignore` 里）。它是变更日志 + 路线图，
  从磁盘读，**永远别 `git add` 它**。
- **绝不提交受版权保护的素材或测试夹具。** 真视频样本一律 `.gitignore`，测试运行时按
  环境变量路径读、缺样本打 `SKIP …` 并通过 —— `include_bytes!` 一个仓库里没有的文件
  会让每个新 clone 的 `cargo test` 直接崩。
- **本地调试残留不要留在仓库里。** 排查期随手写的东西（读 `/tmp` 里某个中间文件、
  写死某次实验的路径、只服务一次定位的 example 或脚本）在排查结束后**立刻删掉**。
  它们既漏信息（路径、文件名、量出来的数字都是「夹具出处」的另一种形式），
  也是给读者看的噪音。要留就留成**有正经用法、参数全从命令行来**的小工具。
- **绝不在任何会进提交的地方点名真实的来源产品**（提交信息、代码注释、测试名）。
  具体片名/文件名/从某个样本量出来的数字 = **夹具出处**，等于告诉读者这来自哪个商业产品，
  而 `git log` 和源码是公开且永久的。只描述**形状**（「一个真 WMV 样本」
  「一个只装脚本文本的小补丁包」），具体信息留在本地未跟踪的 `TODO.md`。
  上游工具已公布的格式名不受限（编解码器名、容器名照常写）。
- **🚫 代理永远不能自主 `git push`。** 这是本仓最硬的一条规矩，没有例外：
  - 代理**只**能在本地 `git commit`；**任何**形式的推送（含 `git push -f`、含
    「用户看起来已经授权」的情形）都必须**先停下来**，把要推的内容摆出来，
    等**人明确复审并针对这一次单独授权**。
  - **一次授权只覆盖一次。** 用户说过一次「可以推了」，不代表以后都可以 ——
    下一次推送要**重新**授权。
  - 复审是**人**的动作：代理不能用「测试过了」「扫描干净了」来代替人工复审。
    代理可以（也应该）把扫描结果摆出来**帮助**复审，但结论由人下。
  - 违反这条 = 本仓最严重的错误，比写错代码严重 —— 代码错了可以改，推出去的东西
    进了 `git log` 和公开历史就收不回来。
- **一张派发表只有一份真源。** 重复的 `match`/`when` 分支一定会漂 —— 加格式时只改那一处，
  别的地方转发。
- **真样本 + 独立 oracle 才算验证过。** 自己造、自己读的往返测试**必要但不充分** ——
  它和它编码的假设会一起错。对着 `ffmpeg`/`ffprobe` 逐样本比（解码结果、帧数、时间戳），
  并且优先「**字节相同**」而不是「没报错」。

## 命令

```bash
cargo test --workspace -j 4     # 单元 + 不依赖样本的集成测试（缺样本的会打 `SKIP` 并通过）
cargo build -j 4                # 并发**限 4** —— 见下一节「构建资源纪律」

cargo run -p v2v-cli -- convert in.wmv -o out/              # .wmv：MP43 + 音频直通
cargo run -p v2v-cli -- convert --mp4 -g 12 in.wmv -o out/  # .mp4：mp4v + AAC

# VC-1 逐帧逐像素对账（最吃时间的一条；V2V_MAX_FRAMES 可先限量跑）
V2V_ASF_SAMPLE=/path/x.wmv cargo test -p v2v-vc1 --release --test frames -- --nocapture
```

**带真样本的对账测试**按**环境变量**找样本（样本一律 `.gitignore`，缺了就 `SKIP`）：

| 变量                       | 用途                                              |
| ------------------------ | ----------------------------------------------- |
| `V2V_ASF_SAMPLE`         | 视频逐帧对 `ffmpeg`（msmpeg4 / avi / vc1 / cli；多个样本用 `:` 分隔） |
| `V2V_WMA_SAMPLE`         | WMA 解码出的 PCM 对 `ffmpeg`（wma）                    |
| `V2V_AUDIO_SAMPLE`       | 音频直通整文件对账（cli，标了 `#[ignore]`）                    |
| `V2V_MP4_REF` / `V2V_MP4_PCM` | MP4 封装逐 box 对 `ffmpeg` 产的参考件（mp4）              |
| `V2V_MP4_OUT` / `V2V_MP4_OUT_AV` | 上面那两条把我们的产物**落到哪个路径**（不设就落 `/tmp/v2v_ours*.mp4`） |
| `V2V_BENCH_SAMPLE`       | 解码速度对表（msmpeg4 / vc1 的 `decode_bench`，`:` 分隔多个样本；不设就 SKIP） |
| `V2V_BENCH_FRAMES` / `V2V_BENCH_REPS` | 上面那条的「只比前 N 帧」/「取几次里最快」（默认全解 / 3 次）        |
| `V2V_MAX_FRAMES` / `V2V_PFRAME_MAP` | 限量跑 / 逐帧定位（msmpeg4、vc1 调试用）                    |
| `V2V_FRAME_LOG`          | 逐帧打印「用掉多少位 / 差多少像素」（vc1）                      |
| `V2V_FUZZ_ROUNDS`        | 模糊测试的轮数（见下；不设就用各局自己的默认）                          |

⚠️ **`V2V_ASF_SAMPLE` 的「多个样本用 `:` 分隔」只管 msmpeg4 / avi / vc1 / cli**；
`asf` 的 oracle 把整串当一个路径（`is_file()` 判掉）—— 会**静默 SKIP**。
要多样本覆盖就**逐个样本跑**，别把 `:` 列表喂给全部 crate。

⚠️ **SKIP 行是被 cargo 捕获的**：不 `--nocapture` 就看不见，于是「全绿」会被读成
「都真跑了」。报证据时要连**「跑了几条 / 跳了几条」**一起报，或者看**用时**
（`finished in 0.00s` 的「ok」多半是早退）。

```bash
# 例：带样本跑全套；最重的是 #[ignore] 那一层
V2V_ASF_SAMPLE='/path/a.wmv:/path/b.asf' cargo test --workspace -j 4 -- --ignored
```

⚠️ **「`cargo test` 全绿」≠「对账过了」** —— 没设环境变量时那些对账测试只是 **SKIP 了**。
动了编解码 / 容器，必须自己带上样本再跑一遍（见「完成任务前」）。

### 变异模糊测试（`crates/*/tests/fuzz.rs`）

**零样本、零 ffmpeg、纯内存** —— 所以它在 CI 里也真跑，不是「SKIP 了」。
六个目标各一局：MP43 解码 / WMA 解码 / ASF 解封装 / AVI 解封装 / MP4 封装 / **VC-1 帧头**。
判据是「**允许 `Err`，禁止 `panic`**」—— 解码器吃的是别人的字节，
一条 panic 会把整个转换流程带走。

```bash
cargo test --workspace -j 4                          # 轻局：几百轮，秒级
V2V_FUZZ_ROUNDS=30000 cargo test --release -j 4 --workspace -- --ignored --nocapture   # 重局
```

⚠️ **两个坑，改这些测试之前先读**（都踩过）：

- **底本会退化，退化的模糊测试是假绿。** MP43 那局一开始用平缓渐变当底本，
  编出来的 P 帧只有 3 字节（整帧一个 skip 宏块）——「几百轮全绿」其实只测了帧头。
  所以每局都有「底本质量断言」：底本自己先解一遍、通路先验一遍。
- **「没报错」不等于「解进去了」。** `decode_superframe` 对攒位的包返回 `Ok(vec![])`，
  于是「几千轮没报错」可能是「几千轮什么都没解」。所以 WMA 那局数的是
  **真的产出了多少帧**，不是「多少轮没报错」。

变异算子只有一份（`crates/msmpeg4/tests/common/mutate.rs`），`avi` / `asf` / `wma` / `mp4`
用 `#[path]` 引它。**`vc1` 那局是例外、也是有意为之**：它要的底本不是「一段现成字节」，
而是「先按语法**写出**一个像样的帧头、再变异」—— 形状不同，硬套那份算子只会得到一个
测不到深处的底本。真要合，就先把「造底本」与「变异」拆成两件事再合。

## 🖥️ 构建资源纪律（**动手前先看表** —— 这台机器很吃紧）

开发机是 **8 核 / 16 GB**，而且**经常有多个 AI 会话同时开在同一个仓库上**。
实测过一次：多会话并行 + 反复全量 `cargo build --release`，把 load average 顶到 **28**
（8 核机器 = 300% 超载），机器卡到几乎不能用。**代理必须自己盯着，别等人喊。**

**跑任何 `cargo` 命令之前和之中，先看这三张表**：

```bash
uptime                      # load average（1/5/15 分钟）—— 与核数比
vm_stat | head -8           # free / 换页；swapouts 一直涨 = 内存不够
df -h /System/Volumes/Data  # 磁盘 <10% 可用会让 macOS 整体变卡（换页/Spotlight/日志）
```

> ⚠️ 本环境里 `ps aux` / `top` **跑不了**（沙箱），但 `uptime` 可以 —— 别在 `top` 上浪费时间。

规矩：

- **load average 超过核数（8）就不要开新构建**；先等它降下来，或者把并发的会话停掉。
- **能 debug 就别 release。** `--release` 的优化（尤其 LTO）会把所有核拉满几分钟；
  只为跑测试、看报错、量体积以外的东西，debug 足够。
- **限并发**：`cargo build -j 4` / `cargo test -j 4`。不要让它吃满所有核。
- **不要多个会话同时跑同一个仓库的全量 `cargo test --workspace`。** 既费电，
  也会互相抢 `target/` 的锁。同仓库并行请走 `git worktree` 换目录（各自的 `target/` 独立，
  代价是重复编译与双份磁盘 —— 这个权衡要主动跟用户说）。
- **长命令放后台**，别让一次 `cargo build --release` 阻塞住整个会话。
- **发现自己把机器压垮了就立刻收手**：停掉在跑的重活，把「已经验到什么程度」如实报给用户，
  别硬撑着跑完。
- 报进度时**带上 load / 磁盘数字**，用户才知道还要不要继续等。

## 架构不变量

**这几条是「每段可查、可单独替换」得以成立的前提 —— 动它们要当心。**

1. **容器解封装只经 `v2v-container` 的类型对外**（`Stream` / `Sample` / `AudioFormat` /
   `VideoFormat`）。上层（解码器、`v2v convert`）**不认识容器** —— 加一种容器 = 加一个
   「字节 → Stream / Sample」的实现，上层不动。
2. **零第三方依赖。** 所有 crate 只依赖本仓内部包。要引 crates.io 上的东西，先过许可证
   （LGPL 兼容：Apache-2.0 / MIT / BSD；MPL 这类 copyleft 一律不引），并说清现有代码
   为什么做不到。
3. **`v2v-msmpeg4` 兼共享数值内核**：`BitReader` / `BitWriter`（MSB 优先）/ FDCT / IDCT /
   VLC 表 / `Error` 被 `v2v-mp4v`、`v2v-aac`、`v2v-vc1` 复用。**不要复刻第二份** ——
   会漂的就是第二份。改这些共享件要**同时**跑 mp4v / aac / vc1 的对账测试。
   （`vc1` 只借位读取器与前缀码解码，**它自己的逆变换是独立的** —— VC-1 的 8×8 核
   与 H.263 那套不是一回事，别为了「统一」把两套混成一套。）
4. **码表只有一份真源**：`crates/*/src/tables.rs` 是唯一落点，生成与自检脚本在
   `tools/`（`gen_tables.py` / `gen_aac_tables.py` / `gen_wma_tables.py` / `check_tables.py`）。
   加格式只改那一处，别的地方转发。
5. **样本与「夹具出处」永不入库。** 真样本走环境变量 + 缺样本 `SKIP`；**提交信息、源码
   注释、测试名**里不许点名来源产品、不许写样本文件名。**某个样本的规模**（帧数 / 分辨率 /
   单帧字节数）**别写进代码或断言** —— 换个样本就失效，而且它本身就是出处（v0.1.1 清过一批）。
   质量 / 体积指标（PSNR、码率、压缩比）写 README / TODO 没问题，那是测量结果不是出处。
6. **正确性由「真样本 + 外部 oracle」钉住**：解码逐像素对 `ffmpeg`、封装逐 box 对
   `ffprobe`、编出来的流交回 `ffmpeg` 解。自造往返**必要但不充分**，不能单独当判据。

## 已经踩过的坑（从 UsefulUnpack 的视频链路带过来，直接相关）

- **ffmpeg 构建**：
  - `--disable-everything` **管不到** `avfilter` / `avdevice` / `swscale` / `postproc` ——
    这四个是独立子库，要逐个点名关掉（漏了 `swscale` 光它 ~1.1 MB/ABI）。
  - `--ar` / `--ranlib` 必须显式指向 NDK 的 `llvm-*`。用宿主机 Apple `ar` 会把 ELF 成员
    **静默丢弃**：`make` 成功，产出 96 字节的空 `.a`。
  - **`configure` 对拼错的组件名静默忽略**（不报错、不编进去）。名字要用 configure 的名字，
    不是 `ffmpeg -demuxers` 打印的运行时名 —— 两者**不一样**（例：MPEG-PS 是 `mpegps`）。
    改完配置要**逐个 grep 注册表**核对（`codec_list.c` / `muxer_list.c` / `demuxer_list.c`）。
  - 体积只能按**最终 `.so`** 量：`-ffunction-sections -fdata-sections` +
    `-Wl,--gc-sections` 之后才是真实的；`.a` / `.o` 只能当排序用。
- **时间戳**：PTS 只允许前进，**首帧单独判** —— 拿 `last_pts` 的初值 0 去比，会把源里
  合法的**首帧 PTS=0** 判成「回退」，把它推到一帧之后，下一帧又相等 → **级联，整条轨平移一帧**。
  复用器的 `muxdelay`（MPEG-PS 从 0.54 s、TS 从 1.44 s 起，AVI/FLV 从 0）要把整条轨归零，
  否则和另一条轨整整错开这一段。
- **编码器收尾**：EOS 之后编码器是**异步**吐存货的，非阻塞轮询**一次等于没等**
  （实测丢 1–5 帧）。收尾要轮询到看见 EOS 输出标志或超时。
- **`AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG`**：编码器会把 csd（H.264 的 SPS/PPS、AAC 的 ASC）
  单独发一个带该旗标的 buffer，**尺寸也 > 0** —— 只判 `size > 0` 会把它当普通样本写进封装器
  （多一个样本、PTS 错位）。
- **`elst.segment_duration` 是 movie 时间轴，`media_time` 是媒体时间轴** —— 两个单位不一样。
  当帧数用之前要按 `mvhd` / `mdhd` 的 timescale 缩放（ffmpeg 恰好让它们相等，所以容易漏）。
- **只能出 WMV7/WMV8**：ffmpeg 里**没有 VC-1/wmv3 编码器**（只有解码器）。想要 WMV9 得另找。
- **Android 侧**：`MediaCodec` 没有 WMV/VC-1 编码器、`MediaMuxer` 不认 ASF ——
  所以「往 wmv 走」只能全程 ffmpeg。

## 完成任务前

1. `cargo test --workspace -j 4` 绿，**并且**编译**无警告**（本仓现在是零警告）。
2. 动了编解码 / 容器：对着**真样本 + `ffmpeg` / `ffprobe`** 比一遍 —— 合成往返**不算数**，
   而且**没设环境变量时的「全绿」也不算**（那些测试跳过了）。优先「字节 / 像素相同」
   而不是「没报错」。
3. 清调试残留：`grep -rn 'V2V_' crates/*/src` 只应剩下测试里的样本变量；排查期随手写的
   example / 脚本 / 写死路径已经删掉（AGENTS.md「本地调试残留」那条）。
4. 判据、量出来的数字、试过又撤掉的候选 —— 写 `TODO.md`，**不进提交信息**。
