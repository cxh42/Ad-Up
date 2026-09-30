# data/：所有数据

这个目录不进 git（除了本说明）。代码里的路径都定义在 `adup/paths.py`。2026-09-30 按用途重新整理过（旧名见文末）。

## 一眼看懂

数据按"在流程里干什么"分成六个目录。只有 `pairs/` 是训练和评测直接用的数据集，其余是做数据集的原料和辅助：

```
data/
  sources/     ① 原料：高质量素材，用来做 GT                          38 GB
  real_ads/    ② 参照：真实广告，只用来测量、校准、评测，从不训练       3.2 GB
  stats/       ③ 测量表：从 ① ② 测出来、生成数据时要读的统计           10 MB
  pairs/       ④ 成品：我们生成的 (GT, LQ) 配对数据集                  65 GB
  eval_sets/   ⑤ 公开评测集：别人的低质量视频，只评测                  3.5 GB
  assets/         字体、人脸检测模型                                    81 MB
```

流程：`sources/`（素材）+ `stats/`（真实广告的统计）→ 编排和生成 → `pairs/`（配对）；
模型在 `pairs/` 上训练和测试，另在 `real_ads/` 的 test 份和 `eval_sets/` 上做真实视频评测。

**要用数据时看这里**：

| 我要…… | 用哪个 |
|---|---|
| 训练 / 微调模型 | `pairs/v7_train/` |
| 调参、选模型 | `pairs/v7_dev/`，真实视频用 `real_ads/` 的 dev 份 |
| 报最终结果 | `pairs/v7_test/`、`pairs/v7_test_heavy/`，真实视频用 `eval_sets/real_ugc_v1/`（切自 `real_ads/` 的 test 份，人评也用它） |
| 给人看例子 | `outputs/figures/examples/index.html`（由 `pairs/examples/` 生成） |

真实广告属于哪一份，看 `stats/real_ads/splits.csv`，文件列表在 `stats/real_ads/groups/<份>_{hd,sd}.txt`。

## ④ pairs/：生成的配对数据集

| 目录 | 内容 | 大小 |
|---|---|---|
| `v7_train/` | 训练集：1,119 条广告 × 2 个 LQ = 2,238 对。LQ 短边 360 / 540 / 720，10% 重退化、90% 按真实广告校准的日常退化。用了全部 677 条训练份素材，每条素材最多出现 3 次 | 41 GB |
| `v7_dev/` | 开发集：50 条广告，网格：每条 8 个 LQ（270 / 360 / 540 / 720p × H.264 / VP9，校准版退化） | 4.4 GB |
| `v7_test/` | 测试集：100 条广告，网格同上 | 8.0 GB |
| `v7_test_heavy/` | 测试集前 50 条的同一 GT，网格全部用重退化（"看不清"的情况） | 9.8 GB |
| `examples/` | 展示用的小样本：`main/`（36 条，覆盖各种风格和版式）、`heavy/`（其中 9 条的重退化版本）、`grid/`（3 条网格） | 1.6 GB |

train / dev / test 按**源视频**划分：同一个源视频的片段、照片、录屏只会出现在一份里，测试集的素材从不进训练集。

每个数据集目录的结构：

```
v7_train/
  specs.jsonl          每条广告的编排（用哪些素材、怎么剪、画幅、风格、文字……），由 adup/ugc/director.py 写入
  config.yaml          生成时用的全部参数（configs/pairs/v7.yaml 的副本）
  ugc_<素材 id>_<k>/   一条广告
    gt.mp4             GT：1080×1920 或 1920×1080
    lq_0.mp4、lq_1.mp4 LQ（网格数据集里是 lq_<短边>_<编码>.mp4，共 8 个）
    mask.mkv           文字叠加层的逐帧 alpha（算文字区域指标用）
    meta.json          这条广告的一切：素材来源、镜头边界、文字位置和内容、每个 LQ 的退化参数
```

生成方式：`configs/batches/v7_local.yaml` 用 `python -m adup.batch run` 批量生成（本机 2026-09-30，单进程约 12 小时），
`python -m adup.batch status configs/batches/v7_local.yaml` 看进度；日志在 `outputs/logs/batch_v7_local/`。

## ① sources/：高质量素材（GT 的原料）

| 目录 | 内容 | 能用的 | 许可 |
|---|---|---|---|
| `ultravideo/4k/`、`8k/` | UltraVideo 片段，`<类别>/<片段 id>.mp4`，4K 719 条、8K 228 条；`ultravideo/` 下的 `short.csv`、`zip_index.json` 是全部片段的目录和下载索引 | 4K 541 条、8K 170 条 | CC-BY-4.0，仅限非商业研究 |
| `kwaivir/clips/` | KwaiVIR（NTIRE 2026）训练集里的 200 条高清竖屏短视频，1080×1920 | 190 条 | 仅限研究 |
| `unsplash_lite/` | Unsplash Lite 照片元数据（`*.tsv000`）；`images/` 是生成时按需下载的照片 | 1,357 张广告主题照片 | 允许内部商用训练 |
| `ui_screens/` | 合成的手机 App 界面长截图（录屏镜头用），12 张 | 12 张 | 自有 |

每个视频素材目录里有三张表，编排时都要读：
- `manifest.csv`：素材清单（下载脚本写入）；
- `gate_1080.csv`：1080p 预筛结果，`pass` 列为真的才能用（`adup/sources/gate.py`）；
- `source_text.csv`：素材本身是否带字幕 / 水印，带字的不再加字幕（`adup/sources/source_text.py`）。

"能用的"一列就是通过预筛的数量，共 901 条，按源视频分成训练 677 / 开发 107 / 测试 117。没通过的素材留着不删，预筛表里有记录。

## ② real_ads/：真实广告

`<采集脚本>/<采集日期>/`，由 `adup/real_ads/` 下同名脚本写入。只用于测量、校准和评测，从不训练。

| 目录 | 内容 |
|---|---|
| `meta_adlib_web/20260924/` | Meta 广告库视频广告，164 条，只有 720p |
| `meta_adlib_web/20260928/` | Meta 广告库视频广告，64 条，同一条广告的 720p 和 360p 都有（`videos_sd/` 是 360p） |
| `tiktok_topads/20260923/`、`20260924/` | TikTok Creative Center Top Ads，55 + 42 条，576–720p |

每个日期目录里：`videos/<行业或关键词>/*.mp4`，`summary.csv`（元数据和文案），`shots/<广告 id>.json`（镜头边界）。
跨天重复抓到的广告在 `stats/real_ads/splits.csv` 里合并，去重后 248 条，按广告 id 分成 calibration 61 / dev 62 / test 125
（其中有 360p 版本的：18 / 15 / 29）。calibration 只用来校准退化，test 只用来报最终结果和人评。

## ③ stats/：测量表

生成数据时要读这些表，所以放在数据这边；在服务器上生成时要一起带过去。

| 文件 | 内容 | 谁写 | 谁读 |
|---|---|---|---|
| `real_ads/ugc_look.csv` | 真实广告逐镜头的抖动、平移、景深、色彩、人脸 | `analysis/ugc_look.py` | 编排（手持抖动目标） |
| `real_ads/shots.csv` | 真实广告的镜头长度 | `analysis/shots.py --table` | 编排（剪辑节奏） |
| `real_ads/ad_anatomy.csv` | 48 条真实广告的人工标注：风格、画面形式、文字和图形元素 | 人工 | 设计风格和文字比例 |
| `real_ads/content_ads.csv` | 真实广告每 2 秒一帧的内容标签 | `analysis/ugc_content.py ads` | 主题覆盖统计 |
| `real_ads/text_overlay.csv` | 真实广告的画面文字（OCR） | `analysis/text_overlay.py` | 设计文字样式 |
| `real_ads/degradation.csv`、`quality.csv` | 真实广告的退化指标和质量分 | `analysis/degradation_stats.py`、`quality.py` | 校准（`analysis/compare.py`） |
| `real_ads/meta_streams.csv` | Meta 广告的码流信息 | 一次性分析 | 文档 |
| `real_ads/splits.csv`、`groups/` | 真实广告的 calibration / dev / test 划分和各份文件列表 | `real_ads/splits.py` | 校准、基准的真实赛道 |
| `sources/content_clips.csv` | 已下载素材的主题和画面形式（CLIP 看画面） | `ugc_content.py clips` | 编排（按主题抽素材） |
| `sources/content_pool.csv` | UltraVideo 全部片段的主题（按文字描述） | `ugc_content.py pool` | 编排 |
| `sources/content_unsplash.csv` | Unsplash 照片的主题和尺寸 | `ugc_content.py stills` | 编排（照片镜头、卡片） |
| `sources/content_coverage.csv` | 每个主题在真实广告中的占比 vs 素材池里的数量 | `ugc_content.py coverage` | 编排（按主题加权） |
| `eval_sets/` | VideoLQ 和 HQ-VSR（DOVE 原训练集，视频已删）的退化指标和质量分，作参照 | 同 real_ads | 校准对照 |

合成数据的指标不放这里，放在 `outputs/calibration/`。

## ⑤ eval_sets/：公开评测集

| 目录 | 内容 | 用途 |
|---|---|---|
| `KwaiVIR/wild/` | 48 条快手野外低质量短视频（1080×1920） | 真实 UGC 视频的补充评测 |
| `KwaiVIR/val_input/`、`test_data/` | KwaiVIR 比赛的验证、测试输入 | 同上 |
| `KwaiVIR/shots/` | 野外视频的镜头边界 | 按镜头推理 |
| `VideoLQ/` | 真实世界视频超分基准（DOVE 论文用过） | 核对我们跑 DOVE 的结果和论文一致 |
| `real_ugc_v1/` | **我们的真实低质量测试集**：从 `real_ads/` 的 test 份切出的 36 段 5 秒片段（28 段 Meta 360p + 8 段 TikTok 576p / 360p）；`inputs/` 给方法，`anchors_hd/` 是 Meta 720p 同帧参照，`segments.csv` 记录来源和起止帧 | 人评、真实赛道的自动指标（`docs/benchmark.md` 第 5 节） |

KwaiVIR 的 200 条高清片段是 GT 素材，已移到 `sources/kwaivir/clips/`；比赛自带的合成 LQ 用不上，已删。

## assets/

`fonts/`（字幕用的 OFL 开源字体和 Noto Color Emoji，`python -m adup.ugc.fonts` 下载），
`models/face_detection_yunet_2023mar.onnx`（YuNet 人脸检测）、`models/RealESRGAN_x4plus.pth`（基准测试用）。

## 旧名对照（2026-09-30 整理前）

| 以前 | 现在 |
|---|---|
| `hq/` | `sources/` |
| `benchmarks/KwaiVIR/train/synthetic/HQ-synthetic*/` | `sources/kwaivir/clips/` |
| `benchmarks/` | `eval_sets/`（`KwaiVIR/train/wild/` → `KwaiVIR/wild/`） |
| `pairs/ugc_v7_{train,dev,test,test_heavy}/` | `pairs/v7_{train,dev,test,test_heavy}/` |
| `pairs/ugc_v7_examples{,_heavy,_grid}/` | `pairs/examples/{main,heavy,grid}/` |
| `stats/hq/`、`stats/benchmarks/` | `stats/sources/`、`stats/eval_sets/` |

所有表格、编排文件和 meta.json 里的路径都已同步改过。代码包 `adup/hq/` 同时改名为 `adup/sources/`。
