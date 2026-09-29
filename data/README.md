# data/：所有数据

这个目录不进 git（除了本说明）。每个子目录只放一类东西，按流程顺序：

```
real_ads/      ① 真实广告：目标 LQ 域，只做统计和评测，不做训练
hq/            ② 高质量素材（在 1080p 下真实清晰）：GT 的来源
stats/         ③ 测量表：从 ① 和 ② 测出来、配对流程运行时要读的统计
pairs/         ④ 生成的 (GT, LQ) 配对数据集
benchmarks/       公开数据集：评测和参考
assets/           字体、人脸检测模型
```

代码里的路径都定义在 `adup/paths.py`。数量是本机 2026-09-28 的情况。

## real_ads/：真实广告

`<采集脚本>/<采集日期>/`，由 `adup/real_ads/` 下同名脚本写入。

| 目录 | 内容 |
|---|---|
| `tiktok_topads/20260923/`、`20260924/` | TikTok Creative Center Top Ads，55 + 42 条（576–720p）；`videos_1080p/` 是其中 18 条的 1080p 版本 |
| `meta_adlib_web/20260924/` | Meta 广告库视频广告，164 条（只有 720p） |
| `meta_adlib_web/20260928/` | Meta 广告库视频广告，64 条，720p 和 360p 两个版本都有（跨天重复的广告在 `splits.csv` 里合并） |

每个日期目录里：`videos/<行业或关键词>/*.mp4`（Meta 广告库另有 `videos_sd/`，同一条广告的 360p 版本，2026-09-27 起的抓取才有），`summary.csv`（每条广告的元数据和文案，字幕文案也从这里取），
`shots/<广告 id>.json`（镜头边界，`adup/analysis/shots.py` 写入）。

## hq/：高质量素材

| 目录 | 内容 | 许可 |
|---|---|---|
| `ultravideo/` | `short.csv`（UltraVideo 全部片段的目录）和 `zip_index.json`（片段在远程 zip 里的位置），4K 和 8K 共用 | |
| `ultravideo/4k/` | 719 条 4K 片段，`<类别>/<clip_id>.mp4` | CC-BY-4.0，仅限非商业研究 |
| `ultravideo/8k/` | 228 条 8K 片段 | 同上 |
| `unsplash_lite/` | Unsplash Lite 的照片元数据（`*.tsv000`）；`images/` 是渲染时按需下载的照片 | 允许内部商用训练 |
| `ui_screens/` | 合成的手机 App 界面长截图（录屏镜头的 GT），12 张 | 自有 |
| `kwaivir/` | 只有 `manifest.csv` 和 `gate_1080.csv`：指向 `benchmarks/KwaiVIR/train/synthetic/HQ-synthetic*/` 的 200 条原生竖屏 1080×1920 短视频，类别来自快手标签 | NTIRE 2026 比赛数据，仅限研究 |

每个视频目录里有 `manifest.csv`（素材清单，下载脚本写入）和 `gate_<GT 短边>.csv`（预筛结果，`adup/hq/gate.py`
写入）。v7 用 `gate_1080.csv`：都已跑完，KwaiVIR 190 / 200、UltraVideo 4K 541 / 719、8K 170 / 228 通过（任一方向）。
`source_text.csv`（`adup/hq/source_text.py`）标出本身带字幕的片段，编排时用 `--source-text` 读入：都已跑完：KwaiVIR 103 / 200、UltraVideo 4K 108 / 541、8K 22 / 170 带字（只查通过预筛的）。

## stats/：测量表

配对流程运行时要读这些表，所以放在数据这边；服务器上跑批量生成时要一起带过去。

| 文件 | 内容 | 写入 | 读取 |
|---|---|---|---|
| `real_ads/ugc_look.csv` | 真实广告逐镜头的抖动、平移、景深、色彩、人脸 | `analysis/ugc_look.py` | `ugc/director.py`（手持抖动目标） |
| `real_ads/shots.csv` | 真实广告的镜头长度（46 条 TikTok） | `analysis/shots.py --table` | `ugc/director.py`（切镜节奏） |
| `real_ads/content_ads.csv` | 真实广告每 2 秒一帧的内容标签 | `analysis/ugc_content.py ads` | `ugc_content.py coverage` |
| `real_ads/text_overlay.csv` | 真实广告的画面文字（OCR） | `analysis/text_overlay.py` | 设计字幕样式时参考 |
| `real_ads/degradation.csv`、`quality.csv` | 真实广告的退化指标和质量分 | `analysis/degradation_stats.py`、`quality.py` | `analysis/compare.py`（校准） |
| `real_ads/meta_streams.csv` | Meta 广告的码流信息（编码器、码率） | 一次性分析 | 文档 |
| `real_ads/groups/` | 上面这些测量用到的广告列表 | | |
| `real_ads/ad_anatomy.csv` | 48 条真实广告的人工标注：风格、画面形式、文字和图形元素（`docs/ugc_dataset.md` 1.3 节） | 人工 | 设计 `ugc.director.style` 和 `ugc.text.by_style` 的比例 |
| `real_ads/splits.csv` | 每条真实广告属于 calibration / dev / test 哪一份（按广告 id 哈希，固定不变） | `real_ads/splits.py` | 校准只用 calibration，测试只用 test |
| `real_ads/groups/<split>_{hd,sd}.txt` | 每一份的真实广告文件列表（HD：TikTok 和 Meta 720p；SD：Meta 360p） | `real_ads/splits.py` | 校准、基准的真实赛道 |
| `hq/content_pool.csv` | UltraVideo 全部片段的主题（按文字描述） | `ugc_content.py pool` | `ugc/director.py` |
| `hq/content_clips.csv` | 已下载片段的主题（CLIP 看画面） | `ugc_content.py clips` | `ugc/director.py` |
| `hq/content_unsplash.csv` | Unsplash 照片（短边 ≥1440）的主题和横竖 | `ugc_content.py stills` | `ugc/director.py`（照片镜头）、`hq/ui_screens.py` |
| `hq/content_coverage.csv` | 每个主题在真实广告中的占比 vs 素材池里的数量 | `ugc_content.py coverage` | `ugc/director.py`（按主题加权抽素材） |
| `benchmarks/` | VideoLQ、HQ-VSR 的退化指标和质量分，作参照（HQ-VSR 原视频已删，统计保留） | 同 real_ads | `analysis/compare.py` |

合成数据的指标不放这里，放在 `outputs/calibration/`。

## pairs/：配对数据集

一个数据集一个目录，由 `adup/ugc/director.py`（编排）和 `adup/make_pairs.py`（生成）写入：

```
<数据集>/
  specs.jsonl            每条广告的编排（镜头、来源、画幅、转场……）
  config.yaml            生成时用的参数（configs/pairs/<版本>.yaml 的副本）
  <广告 id>/
    gt.mp4  lq_0.mp4  lq_1.mp4     整段（v7：GT 1920x1080 / 1080x1920，LQ 短边 360 / 540 / 720）
    mask.mkv                       文字叠加层的逐帧 alpha（无损 FFV1 灰度）
    shots/shot_XXX_{gt,lq_k}.mp4   多镜头时的逐镜头版本（逐帧精确、无损）
    meta.json                      每一步的参数、镜头边界和来源、所属划分（train / dev / test）
```

本机的配对数据（展示样例 2026-09-29，单进程、在 `adup.memguard` 下生成，峰值内存 3–6 GB）：

| 目录 | 内容 |
|---|---|
| `data/pairs/ugc_v7_examples/` | 36 条广告 × 2 个 LQ，从训练集编排里按覆盖面挑：两种风格、横竖屏、24 / 25 / 30 fps、12 个广告主题、卡片、分屏、拼图、画中画、手机外框、自带字幕的素材；退化按训练设置（72 个 LQ 里 3 个抽到重退化） |
| `data/pairs/ugc_v7_examples_heavy/` | 其中 9 条的同一 GT，全部用重退化（`--set degrade.rbvsr_prob=1.0`），用来对比日常退化和重退化 |
| `data/pairs/ugc_v7_examples_grid/` | 3 条测试集广告按网格生成：同一 GT 的 8 个 LQ（270 / 360 / 540 / 720p × H.264 / VP9，校准版退化） |
| `data/pairs/ugc_v7_train/` | 用现有全部素材编排的训练集：1,119 条广告 × 2 个 LQ（重退化 10%），每条素材最多出现 3 次 |
| `data/pairs/ugc_v7_dev/`、`ugc_v7_test/` | 50 / 100 条广告，网格：每条 8 个 LQ（270 / 360 / 540 / 720p × H.264 / VP9，校准版退化） |
| `data/pairs/ugc_v7_test_heavy/` | 测试集前 50 条的同一 GT，网格全部用重退化 |

这四份由 `configs/batches/v7_local.yaml` 在本机批量生成（2026-09-30 开始，单进程约 12 小时）：`python -m adup.batch run configs/batches/v7_local.yaml` 运行或续跑，`python -m adup.batch status configs/batches/v7_local.yaml --watch 30` 或 `outputs/logs/batch_v7_local/status.html` 看进度。

展示材料（`adup.analysis.showcase` 生成）在 `outputs/figures/examples/`，打开 `index.html` 可以在一页里看总览图、文字区域对比、
重退化对比、网格对比和每条广告的 GT / LQ 并排视频。本机跑生成类任务都用 `python -m adup.memguard -- <命令>` 包一层：
可用内存低于 10 GB 会自动停掉（任务可续跑）。

## benchmarks/：公开数据集

| 目录 | 内容 | 用途 |
|---|---|---|
| `KwaiVIR/` | NTIRE 2026 快手短视频修复，全部 1080x1920：`train/synthetic/HQ-synthetic*/`（200 条高清片段，作 GT 素材，见 `hq/kwaivir/`）、`train/wild/`（48 条野外低质视频）、`val_input/`、`test_data/`；`shots/` 是野外视频的镜头边界。比赛自带的 `LQ-synthetic*` 用不上，已删 | GT 素材、评测 |
| `VideoLQ/` | 真实世界视频超分基准 | 评测 |

## assets/

`fonts/`（字幕用的 OFL 开源字体和 Noto Color Emoji，`python -m adup.ugc.fonts` 下载），
`models/face_detection_yunet_2023mar.onnx`（YuNet 人脸检测）。
