# v2videos

用 **Rust** 研究「视频之间的编码格式转换」—— 也就是把一种视频（**容器 + 视频编码 + 音频编码**）  
转成另一种，整条链路的每一段都拿出来看看：在纯 Rust 里能做到什么、边界在哪、代价是什么。

> 名字里的 `v2v` = **v**ideo **to** **v**ideo。

## 转换链路

一次转换拆成四段，研究也按这四段走：

```text
         ┌─ 解封装 ──► 解码 ──► 像素域 / 采样域 ──► 编码 ──► 封装 ─┐
输入 ────┤   demux      decode     (YUV / PCM)        encode   mux   ├──── 输出
         └──────────────────────────────────────────────────────────┘
```

中间那层「像素域 / 采样域」是关键的解耦点：两端互相不认识，只认中间那份裸数据。

## 目录

```text
crates/
  container 各容器共用的数据模型（流 / 格式 / 样本）—— 零依赖
  asf      ASF（.wmv / .wma）解封装 + **封装** —— 纯 std
  avi      AVI（RIFF）解封装
  msmpeg4  MS-MPEG-4 v3（MP43）解码（I / P）+ 编码（I / P，含运动估计与帧内/帧间判决）
  wma      WMA v1/v2 音频解码（帧头 / LSP 指数 / 系数 / IMDCT）
  mp4v     MPEG-4 Part 2（mp4v）**编码** —— 目前全 I 帧
  mp4      MP4 / ISOBMFF **封装**（ftyp / mdat / moov，视频 + PCM 音频）
  cli      `v2v` 命令行：**批量转码**（解封装 → 解码 → 编码 → 封装）
tools/
  gen_tables.py   生成 VLC / 反量化表（附自检：前缀无关、Kraft 和、逐码往返）
```

## 构建与测试

```bash
cargo build
cargo test        # 单元测试 + 不依赖样本的集成测试
```

**与 `ffmpeg` / `ffprobe` 的交叉验证**是本仓的判据：解码结果逐像素对、封装字段逐字段对、  
编码出来的流交回 ffmpeg 解。这些测试需要一个真样本，用环境变量指路；**缺样本时打  
`SKIP` 并通过**，所以新 clone 下来 `cargo test` 直接就能跑：

```bash
V2V_ASF_SAMPLE=/path/to/sample.wmv cargo test

# 整段样本逐帧对账（最重的一条，默认标了 #[ignore]）：
V2V_ASF_SAMPLE=/path/to/sample.wmv cargo test --release -- --ignored
```

## 状态

按「一段一段啃」推进。已经落到代码上的：

| 链路分段     | 做到了什么                                                                                                       | 怎么算验证过                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **解封装**  | ASF / WMV（`v2v-asf`）、AVI（`v2v-avi`）                                                                         | 与 `ffprobe` 逐字段对：轨道、尺寸、包数、关键帧数                                                                  |
| **解码**   | MS-MPEG-4 v3（MP43）：I 帧 + P 帧（skip、MV 中值预测、半像素运动补偿、帧间残差）                                                     | 与 `ffmpeg` **逐像素**对：整段样本**每一帧最大差 0**                                                            |
| **编码**   | MP43 **I 帧 + P 帧**：正向 DCT + 量化 + DC / RL 熵编码、运动估计（整像素粗搜 + 半像素细化）、skip 宏块                                    | 编出来的 I+P 序列交给 `ffmpeg` 解，与自己的解码器**逐像素一致**                                                       |
| **编码**   | MPEG-4 Part 2（`v2v-mp4v`）**I 帧 + P 帧**：DC / AC 预测（逐宏块比位数择优）、运动估计（整像素粗搜 + 半像素细化）、中值 MV 预测、skip 宏块、逐宏块帧内/帧间判决 | 编出来的流交给 `ffmpeg` 解，真样本 **44.8 dB**（同 qscale 下 ffmpeg 自己的 mpeg4 是 43.6 dB）；**视频载荷比全 I 帧小 2.3 倍** |
| **音频直通** | 转换时原样搬音频轨（ASF 多流复用）                                                                                         | 真样本整文件（208 秒音乐）：输出音频与源**逐样本 0 差异**（36,700,160 字节完全一致）                                           |
| **音频解码** | WMA v1/v2（`v2v-wma`）→ PCM                                                                                   | 真样本整轨与 ffmpeg 的 PCM **最大差 2**（s16 满量程 32768）；转进 MP4 后**对齐偏移 0 采样点**                             |
| **音频编码** | AAC-LC（`v2v-aac`）：MDCT + 量化 + Huffman 码本 + **短窗**（瞬态）+ 窗口分组 | ffmpeg 解得开；码率命中目标（二分搜 sf 偏移）；质量与 ffmpeg 的 AAC 同基准比（差 2 dB 内）；瞬态素材上**短窗 vs 长窗 12.02 vs 10.49 dB、体积小 29%** |
| **封装**   | ASF 写（`.wmv`，单视频流）                                                                                          | `ffprobe` 认；自己解封装字节一致；ffmpeg 解出来的像素与我们的解码器**逐位一致**                                              |
| **封装**   | MP4 / ISOBMFF 写（`v2v-mp4`，`mp4v` + `sowt` PCM）                                                              | 真样本 mp4v 轨的 `stsz`/`stts`/`stss`/`stco` 与 ffmpeg **逐项相同**、逐帧 MD5 一致；PCM 轨原样往返                   |

**两条输出路**（`v2v convert`，默认 `.wmv`）：

```bash
# .wmv：视频重编码成 MP43，音频**直通**（零损失），会做 PSNR 校验
v2v convert in.wmv -o out/

# .mp4：视频重编码成 MPEG-4 Part 2（-g 给 GOP；默认 0 = 全 I 帧），音频解成 PCM
v2v convert --mp4 -g 12 in.wmv -o out/
```

**为什么 `.mp4` 必须重编码**：MP4 / 3GP 容器**不收 MP43**（`mp4`/`3gp`/`3g2` 三个 muxer 都报  
`codec not currently supported in container`），`mov` 收但是单向街 —— 写出来的东西 ffmpeg  
自己读不回来。WMA 同理。所以 `.wmv → .mp4` 这条路上视频必须过一遍 MPEG-4 Part 2 编码器，  
音频必须解成 PCM（`sowt`）。

**还没做**：MPEG-4 Part 2 的 **B 帧**与 8×8 运动补偿、AAC 的 **M/S 立体声 / PNS**
（TNS 已做，只覆盖长窗）、音频**编码**（WMA 只能解）、Simple Index、其他容器与编解码器。

⚠️ **AAC 编码器在真音乐上还有明显的质量问题**（同一个文件，码率越高反而越差）——
细节与已排除的怀疑写在 `TODO.md`（本地），这是下一步最该啃的一块。

## 命令行

```bash
cargo run --release -p v2v-cli -- convert <输入.wmv>... -o <输出目录> [-q 6] [-g 250] [-n 100]
cargo run --release -p v2v-cli -- convert --mp4 <输入.wmv>... -o <输出目录> [-q 6] [-g 12] [--fps 25]
```

批量：一个输入失败不影响其他。默认（`.wmv`）转完会把输出**解码回来**与源帧比 PSNR；  
`--mp4` 不做这个校验 —— 我们只有 `mp4v` 编码器、没有解码器（那条路的正确性由  
`crates/mp4v/tests/ffmpeg_roundtrip.rs` 钉着）。

输入容器按**魔数**自动认（`ASF` / `AVI`，不看扩展名），视频轨目前只支持 **MP43**；  
别的容器 / codec 会明确报「不支持」，不静默跳过。

## 路线与分支策略

这个项目绕不开一个两难：**要么保 FFI / 结构清晰**（强调可复用性、流畅性、可监督性——  
每一段清晰可查、可单独替换、能当库嵌进别的应用），**要么追齐 ffmpeg 的性能**  
（SIMD / 手写内核 / 定点化，复杂度与可维护性代价都大）。

**我们的取舍：在 Android 上拿到最完整构建是第一优先级，性能暂时让路。** 落到分支上：

- **`main` = 可扩展性**。新能力、新格式、结构改良进 main；性能敏感的实现  
  允许「正确但慢」——先把语义钉对，再谈快。
- **性能改动进按类型归类的专门分支**（同类型改动进同一分支，便于跟进与随时改动）：  
  `perf/encode`（编码提速）、`perf/decode`（解码提速）、`perf/tables`（码表改良）。
- 性能分支**保持短命**：定期 rebase、验证过（输出字节不变）就合回 main——  
  永远不合并的性能优化等于没有。

## 许可证

**LGPL-2.1-or-later**，见 [LICENSE](LICENSE)。

选 LGPL 的**原因很直接：MP43 与 MPEG-4 Part 2 编解码需要的 VLC / 反量化表**  
（`crates/msmpeg4/src/tables.rs`、`crates/mp4v/src/tables.rs`）是从 FFmpeg 的数据文件读出来的  
（同为 LGPL-2.1-or-later）。表本身是 H.263 / MPEG-4 Part 2 的格式定义、与实现无关，  
也可以换数据源重新生成（`tools/gen_tables.py` 就是为此留的）；但与其搞「单文件例外」  
这种容易说错的安排，不如整个仓库统一按 LGPL 走。

**代价**：GPL 侧的编码器（x264/x265）链不了，编码只能走宽松许可的实现或硬件  
（Android MediaCodec）；引第三方 crate 也只挑宽松许可的（Apache-2.0 / MIT / BSD）。  
LGPL 作为**库**被 MIT 的应用依赖是没问题的；反过来（把 LGPL 代码并进 MIT 的库再改许可）不行

**但如果各类创作者（例如MIT/APACHE等开源协议）想要引入我自研的源码（无ffmpeg等依赖的纯自主源码），可开issue等和我单独协商**
