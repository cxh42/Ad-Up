# data/：所有数据

这个目录不进 git（除了本说明）。每个子目录只放一类东西，按流程顺序：

```
real_ads/      ① 真实广告：目标 LQ 域，只做统计和评测，不做训练
hq/            ② 高质量素材（≥2K）：GT 的来源
stats/         ③ 测量表：从 ① 和 ② 测出来、配对流程运行时要读的统计
pairs/         ④ 生成的 (GT, LQ) 配对数据集
benchmarks/       公开数据集：评测和参考
assets/           字体、人脸检测模型
```

代码里的路径都定义在 `adup/paths.py`。数量是本机 2026-09-24 的情况。

## real_ads/：真实广告

`<采集脚本>/<采集日期>/`，由 `adup/real_ads/` 下同名脚本写入。

| 目录 | 内容 |
|---|---|
| `tiktok_topads/20260923/`、`20260924/` | TikTok Creative Center Top Ads，55 + 42 条（576–720p）；`videos_1080p/` 是其中 18 条的 1080p 版本 |
| `meta_adlib_web/20260924/` | Meta 广告库视频广告，164 条 |

每个日期目录里：`videos/<行业或关键词>/*.mp4`，`summary.csv`（每条广告的元数据和文案，字幕文案也从这里取），
`sheets/`（帧缩略图拼图），`shots/<广告 id>.json`（镜头边界，`adup/analysis/shots.py` 写入）。

## hq/：高质量素材

| 目录 | 内容 | 许可 |
|---|---|---|
| `ultravideo/` | `short.csv`（UltraVideo 全部片段的目录）和 `zip_index.json`（片段在远程 zip 里的位置），4K 和 8K 共用 | |
| `ultravideo/4k/` | 719 条 4K 片段，`<类别>/<clip_id>.mp4` | CC-BY-4.0，仅限非商业研究 |
| `ultravideo/8k/` | 228 条 8K 片段，2K+ 竖屏 GT 的主要来源 | 同上 |
| `unsplash_lite/` | Unsplash Lite 的照片元数据（`*.tsv000`）；`images/` 是渲染时按需下载的照片 | 允许内部商用训练 |
| `ui_screens/` | 合成的手机 App 界面长截图（录屏镜头的 GT），12 张 | 自有 |

每个视频目录里有 `manifest.csv`（素材清单，下载脚本写入）和 `gate_<GT 短边>.csv`（预筛结果，`adup/hq/gate.py`
写入；目前只有 8K 跑过，4K 的预筛留给服务器）。

## stats/：测量表

配对流程运行时要读这些表，所以放在数据这边；服务器上跑批量生成时要一起带过去。

| 文件 | 内容 | 写入 | 读取 |
|---|---|---|---|
| `real_ads/ugc_look.csv` | 真实广告逐镜头的抖动、平移、景深、色彩、人脸 | `analysis/ugc_look.py` | `ugc/look.py`（色彩目标）、`ugc/director.py`（抖动目标） |
| `real_ads/shots.csv` | 真实广告的镜头长度（46 条 TikTok） | `analysis/shots.py --table` | `ugc/director.py`（切镜节奏） |
| `real_ads/content_ads.csv` | 真实广告每 2 秒一帧的内容标签 | `analysis/ugc_content.py ads` | `ugc_content.py coverage` |
| `real_ads/text_overlay.csv` | 真实广告的画面文字（OCR） | `analysis/text_overlay.py` | 设计字幕样式时参考 |
| `real_ads/degradation.csv`、`quality.csv` | 真实广告的退化指标和质量分 | `analysis/degradation_stats.py`、`quality.py` | `analysis/compare.py`（校准） |
| `real_ads/meta_streams.csv` | Meta 广告的码流信息（编码器、码率） | 一次性分析 | 文档 |
| `real_ads/groups/` | 上面这些测量用到的广告列表 | | |
| `hq/content_pool.csv` | UltraVideo 全部片段的主题（按文字描述） | `ugc_content.py pool` | `ugc/director.py` |
| `hq/content_clips.csv` | 已下载片段的主题（CLIP 看画面） | `ugc_content.py clips` | `ugc/director.py` |
| `hq/content_unsplash.csv` | Unsplash 照片（短边 ≥1440）的主题和横竖 | `ugc_content.py stills` | `ugc/director.py`（照片镜头）、`hq/ui_screens.py` |
| `hq/content_coverage.csv` | 每个主题在真实广告中的占比 vs 素材池里的数量 | `ugc_content.py coverage` | `ugc/director.py`（按主题加权抽素材） |
| `benchmarks/` | VideoLQ、HQ-VSR 的退化指标和质量分，作参照 | 同 real_ads | `analysis/compare.py` |

合成数据的指标不放这里，放在 `outputs/calibration/`。

## pairs/：配对数据集

一个数据集一个目录，由 `adup/ugc/director.py`（编排）和 `adup/make_pairs.py`（生成）写入：

```
<数据集>/
  specs.jsonl            每条广告的编排（镜头、来源、画幅、转场……）
  config.yaml            生成时用的参数（configs/pairs/<版本>.yaml 的副本）
  <广告 id>/
    gt.mp4  lq_0.mp4  lq_1.mp4     整段
    shots/shot_XXX_{gt,lq_k}.mp4   多镜头时的逐镜头版本（逐帧精确、无损）
    meta.json                      每一步的参数、镜头边界和来源
```

| 目录 | 内容 |
|---|---|
| `ugc_v5_calib/` | 当前 v5 流程，30 条广告 × 2 个 LQ，用于校准 |
| `ugc_v5_preview/` | 当前 v5 流程，8 条广告 × 1 个 LQ，按真实广告主题占比抽取，人工看效果用 |
| `cuttest_2k_x2/` | 4 条多镜头序列，测 DOVE 在镜头切换处的表现 |
| `archive/` | 旧版本：1080p GT 的配对（`uv_p1080_x2*`）、v1–v4 的校准集（`calib_*`）。已被 v5 取代 |

这几个数据集是在引入 `config.yaml` 之前生成的，参数只记录在每条的 `meta.json` 里。

## benchmarks/：公开数据集

| 目录 | 内容 | 用途 |
|---|---|---|
| `KwaiVIR/` | NTIRE 2026 快手短视频修复：`train/`、`val_input/`、`test_data/`，全部 1080x1920；`shots/` 是野外视频的镜头边界 | 评测 |
| `VideoLQ/` | 真实世界视频超分基准 | 评测 |
| `HQ-VSR/` | DOVE 原训练集（约 1080p），不满足 ≥2K，不用作 GT | 参照 |

## assets/

`fonts/`（字幕用的 OFL 开源字体和 Noto Color Emoji，`python -m adup.ugc.fonts` 下载），
`models/face_detection_yunet_2023mar.onnx`（YuNet 人脸检测）。
