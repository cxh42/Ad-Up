# Ad-Up

Ad-Up 做的是面向 UGC 广告的真实世界视频超分：把 576p–720p 的广告放大到 2K/4K。仓库包含三部分：
- **数据采集**：真实广告，作为目标低质量（LQ）域。
- **配对训练数据**：高质量（HQ）素材，加上按真实广告校准过的合成 LQ。
- **模型训练**：微调 DOVE。

## 目录结构

```
adup/                     项目代码（Python 包），在仓库根目录用 `python -m adup.<module>` 运行
  paths.py                所有仓库路径 + 网络代理（ADUP_PROXY，默认 http://127.0.0.1:7897）
  media.py                ffmpeg 读写、裁剪几何、人脸感知的裁剪位置
  collect/                真实广告 = 真实世界 LQ 域（只用于评测）
    tiktok_topads.py      TikTok Creative Center Top Ads -> data/ads/tiktok_topads/<date>/
    meta_adlib.py         通过 AdDownloader 抓 Meta 广告库（需要 META_TOKEN）-> data/ads/meta_adlib/
    meta_adlib_web.py     从公开的 Meta 广告库网页抓视频广告（不需要 token）-> data/ads/meta_adlib_web/<date>/
    contact_sheets.py     为抓到的广告按行业生成帧缩略图拼图
  sources/                用作 GT 的高质量素材
    ultravideo.py         UltraVideo 的人物类和产品展示类 4K/8K 片段，从远程 zip 中逐条取出
    gate.py               GT 质量预筛：曝光、纹理、在 GT 分辨率下的有效分辨率（横竖两种朝向分别打分），并测素材自身运动
    ui_screens.py         合成手机 App 界面长截图（无头 Chrome，4 倍 / 6 倍像素），做录屏镜头的 GT
  ugc/                    把高质量素材做成"像真实 UGC 广告"的 GT（见 docs/ugc_dataset.md）
    director.py           每条素材编成一条广告：镜头切分、跳剪、放大、版式、照片、录屏、开场卡片
    render.py             渲染单个镜头：视频 / 照片 / 版式（模糊填充、分屏、画中画）/ 幻灯片 / 录屏
    camera.py             虚拟手持相机（抖动、漂移、推拉），只用素材多出来的像素
    look.py               手机色彩风格，按真实广告的色彩分布匹配
  shots/
    detect.py             真实视频（广告、KwaiVIR）按镜头切分：原片不动，另存逐镜头无损文件，并记录来源和切法
    compose.py            把单镜头 HQ 片段按真实广告的镜头长度和转场拼成多镜头序列（训练用 5 秒、评测用 15–40 秒）
  degrade/
    pipeline.py           HQ -> (GT, LQ) 配对，逐帧流式处理：横竖多画幅裁剪（只缩小不放大）、镜头转场、文字叠加、
                          拍摄/ISP、剪辑导出、平台转码（H.264/VP9/AV1/HEVC 混合）、二次上传；多镜头序列同时输出
                          整段和逐镜头两个版本；DEFAULT_CONFIG = 校准后的 v4
    overlays.py           在 GT 上烧录滚动字幕、开头标题、贴纸、小字免责声明（按真实广告 OCR 统计设计）
    fonts.py              字幕用的 OFL 开源字体清单，下载到 data/fonts/
  analysis/
    quality.py            DOVER / CLIP-IQA / MUSIQ / 码率 / 有效分辨率
    degradation_stats.py  块效应、噪声、锐化过冲、GOP、重复帧、黑边
    compare.py            真实与合成的分布距离（校准报告）
    text_overlay.py       用 OCR 统计画面文字（字幕、标题、贴纸）的出现率、大小和位置
    ugc_look.py           逐镜头测量"UGC 观感"：手持抖动、平移、景深、色彩、人脸、版式
    ugc_content.py        CLIP / 句向量给真实广告和素材池打内容主题标签
    eval_pairs.py         复原结果对 GT 的逐帧 PSNR / LPIPS，区分切换附近和其他帧
configs/degradation/      早期退化配置（v1），用 --config 传入
training/dove/            我们自己的 DOVE 脚本（上游代码保留在 third_party/DOVE）；infer.py = 按镜头、显存可控的推理
third_party/              上游仓库，以 git submodule 形式引入，不做修改：DOVE（训练/推理）、DOVER
                          （视频质量指标）；权重放在各自目录内且不入库（DOVE/pretrained_models/、
                          DOVER/pretrained_weights/DOVER.pth）
docs/                     ugc_dataset.md（UGC 配对数据集的整体流程与调研）、ugc_degradations.md（真实退化与校准）、
                          hq_sources.md（高质量素材来源）
requirements/             collect.txt（.venv）、ml.txt（.venv-iqa）
data/        （不入库）   ads/  hq/  public/{HQ-VSR,VideoLQ,_archives}  pairs/{<set>,calibration/}  fonts/
outputs/     （不入库）   analysis/（指标 CSV、图、日志）  runs/（训练）
```

## 初始化

```bash
git clone --recurse-submodules <this repo>      # 已有的克隆：git submodule update --init
.venv-iqa/bin/python -m adup.degrade.fonts      # 下载字幕字体到 data/fonts/
```

## 环境

| venv | Python | 用途 |
|---|---|---|
| `.venv` | 3.11 | 广告采集。AdDownloader 要求 Python < 3.12。见 `requirements/collect.txt`。 |
| `.venv-iqa` | 3.11 | HQ 素材获取、退化、指标（torch 2.11、transformers 5）。见 `requirements/ml.txt`。 |
| `.venv-dove` | 3.11 | DOVE 推理和训练（torch 2.8 cu128、diffusers 0.33、transformers 4.51），和 `.venv-iqa` 隔开以免版本冲突。见 `requirements/dove.txt`。 |

## 流程

```bash
# 1. 真实广告（目标 LQ 域）
.venv/bin/python -m adup.collect.tiktok_topads
.venv/bin/python -m adup.collect.meta_adlib_web
.venv/bin/python -m adup.collect.contact_sheets data/ads/<source>/<date>

# 2. HQ 素材
.venv-iqa/bin/python -m adup.sources.ultravideo 25                    # 每类 25 条，每个源视频 1 条
.venv-iqa/bin/python -m adup.sources.ultravideo 300 --per-source 4    # 扩大：约 1000 条 4K
.venv-iqa/bin/python -m adup.sources.ultravideo 1000 --res 8k --per-source 10   # 8K，用于 ×4 -> data/hq/ultravideo_8k/

# 3. GT 预筛 + 拼多镜头序列
.venv-iqa/bin/python -m adup.sources.gate --manifest data/hq/ultravideo/manifest.csv --gt-short 1440 --out data/hq/ultravideo/gate_1440.csv
.venv-iqa/bin/python -m adup.shots.compose --manifest data/hq/ultravideo/manifest.csv --gate data/hq/ultravideo/gate_1440.csv \
    --purpose train --n 300 --out data/hq/sequences/train_2k.jsonl        # --purpose eval：15–40 秒的整条广告

# 4a. UGC 广告配对（推荐）：导演编排 + 渲染；--scale auto 让 LQ 短边在 720/576/540 之间抽取
.venv-iqa/bin/python -m adup.ugc.director --clips data/hq/ultravideo/manifest.csv data/hq/ultravideo_8k/manifest.csv \
    --gate data/hq/ultravideo/gate_1440.csv data/hq/ultravideo_8k/gate_1440.csv --stills outputs/analysis/content_unsplash.csv \
    --screens data/hq/ui_screens/manifest.csv --gt-short 1440 --out data/hq/sequences/ugc_2k.jsonl
.venv-iqa/bin/python -m adup.degrade.pipeline --sequences data/hq/sequences/ugc_2k.jsonl --out data/pairs/ugc_2k --gt-short 1440 --scale auto

# 4b. 其他合成配对（GT 短边 >= 1440 即 2K；--scale 2 时 LQ 短边 720）
.venv-iqa/bin/python -m adup.degrade.pipeline --manifest data/hq/ultravideo/manifest.csv --out data/pairs/uv_2k_x2 --gt-short 1440 --scale 2
.venv-iqa/bin/python -m adup.degrade.pipeline --sequences data/hq/sequences/train_2k.jsonl --out data/pairs/seq_2k_x2 --gt-short 1440 --scale 2

# 5. 真实视频按镜头切分（原片不动）
.venv-iqa/bin/python -m adup.shots.detect data/shots/<set> <videos...>

# 6. DOVE 推理（按镜头；权重放在 third_party/DOVE/pretrained_models/DOVE）
.venv-dove/bin/python training/dove/infer.py --out outputs/runs/dove/<name> --upscale 4 <videos...>

# 7. 校准：分别给合成集和真实广告打分，再比较两者分布
.venv-iqa/bin/python -m adup.analysis.degradation_stats <group> outputs/analysis/degradation.csv <videos...>
.venv-iqa/bin/python -m adup.analysis.quality           <group> outputs/analysis/quality.csv     <videos...>
.venv-iqa/bin/python -m adup.analysis.compare tiktok_720p <group>
.venv-iqa/bin/python -m adup.analysis.text_overlay       <group> outputs/analysis/text_overlay.csv <videos...> --crops <dir>
```

## 数据来源与许可

| 数据 | 位置 | 许可 / 条款 |
|---|---|---|
| TikTok Top Ads | `data/ads/tiktok_topads/` | 广告主的素材。只用于分析和评测。 |
| Meta 广告库 | `data/ads/meta_adlib_web/` | 广告主的素材。只用于分析和评测。 |
| UltraVideo | `data/hq/ultravideo/` | CC-BY-4.0 + **仅限非商业研究**（素材来自 YouTube）。 |
| HQ-VSR（DOVE） | `data/public/HQ-VSR/` | 源自 OpenVid-1M。研究用途。 |
| VideoLQ | `data/public/VideoLQ/` | 研究基准集。 |
| KwaiVIR（NTIRE 2026） | `data/public/KwaiVIR/` | 快手竖屏短视频修复基准，研究用途；1080x1920、6 秒、HEVC。 |

Pexels、Pixabay 和 Mixkit 禁止脚本批量下载，Pexels 和 Pixabay 还明确禁止用于机器学习。
商用训练需要自己拍摄、创作者授权或购买的素材。
