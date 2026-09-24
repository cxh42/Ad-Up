# Ad-Up

Ad-Up 做的是面向 UGC 广告的真实世界视频超分：把 576p–720p 的广告放大到 2K/4K。做法是微调 DOVE，
训练数据是自己造的 (GT, LQ) 配对：
1. 采集真实广告，测出它们长什么样、被怎么退化；
2. 找 2K 及以上（横竖都有）的高质量素材；
3. 把素材做成"像真实 UGC 广告"的 GT（剪辑、手持相机、手机色彩、字幕……）；
4. 按真实广告的退化生成 LQ。

整体流程和调研结论见 `docs/ugc_dataset.md`，数据目录见 `data/README.md`。
给导师的进展汇报（单个 HTML 文件，下载后离线打开）：`docs/progress/`。

## 目录结构

代码按流程分包，每个包对应一个数据目录：

```
adup/                     代码（Python 包），在仓库根目录用 `python -m adup.<模块>` 运行
  paths.py                所有目录和文件位置的唯一定义处 + 网络代理（ADUP_PROXY，默认 http://127.0.0.1:7897）
  config.py               读取数据集参数 configs/pairs/<版本>.yaml；命令行 --set 临时改参数
  media.py                ffmpeg 读写、GT 几何（画幅、裁剪）、人脸检测
  make_pairs.py           生成配对的入口：specs.jsonl -> data/pairs/<数据集>/，串起 ugc/ 和 degrade/ 的各步

  real_ads/            ①  采集真实广告（目标 LQ 域，只做统计和评测）            -> data/real_ads/
    tiktok_topads.py      TikTok Creative Center Top Ads
    meta_adlib_web.py     Meta 广告库网页（不需要 token）
    contact_sheets.py     按行业生成帧缩略图拼图
  analysis/            ②  测量：真实广告的统计 -> data/stats/；合成数据的校准 -> outputs/calibration/
    ugc_look.py           逐镜头的"UGC 观感"：手持抖动、平移、景深、色彩、人脸、版式
    ugc_content.py        内容主题（CLIP / 句向量）：真实广告、素材池、照片，以及主题覆盖表
    shots.py              镜头检测：统计镜头长度；DOVE 按镜头推理也用它
    text_overlay.py       OCR 统计画面文字（字幕、标题、贴纸）
    degradation_stats.py  块效应、噪声、锐化过冲、GOP、重复帧、黑边
    quality.py            DOVER / CLIP-IQA / MUSIQ / 码率 / 有效分辨率
    compare.py            真实与合成的分布距离（校准报告）
    eval_pairs.py         复原结果对 GT 的 PSNR / LPIPS，区分切换附近和其他帧
  hq/                  ③  高质量素材（≥2K）                                     -> data/hq/
    ultravideo.py         UltraVideo 4K/8K 片段，从远程 zip 中逐条取出
    ui_screens.py         合成手机 App 界面长截图（无头 Chrome，4 倍 / 6 倍像素），做录屏镜头的 GT
    gate.py               GT 预筛：曝光、纹理、在 GT 分辨率下的有效分辨率，并测素材自身运动
  ugc/                 ④  把高质量素材做成"像真实 UGC 广告"的 GT
    director.py           编排：每条素材编成一条广告（镜头切分、跳剪、放大、版式、照片、录屏、开场卡片）
                          -> data/pairs/<数据集>/specs.jsonl
    render.py             渲染单个镜头：视频 / 照片 / 版式（模糊填充、分屏、画中画）/ 幻灯片 / 录屏
    camera.py             虚拟手持相机（抖动、漂移、推拉），只用素材多出来的像素
    sequence.py           镜头之间的转场（硬切、叠化、甩镜、黑场、白场）
    look.py               手机色彩风格，按真实广告的色彩分布匹配
    text.py、fonts.py     烧录字幕、标题、贴纸、小字（按真实广告 OCR 统计设计）；OFL 开源字体
  degrade/             ⑤  GT -> LQ 的退化（按真实广告校准）
    capture.py            拍摄 / ISP：运动模糊、虚焦、噪声、降噪美颜、锐化
    platform.py           剪辑导出、平台缩放 + 编码前预处理 + 转码（H.264 / VP9 / AV1 / HEVC）、二次上传
configs/pairs/           每个数据集版本一个参数文件，分 gt / ugc / degrade 三段：v6.yaml（当前，以 Meta 为主）、
                          v5.yaml（已生成的 ugc_v5_* 数据集用的）；更早的版本在 git 历史里
training/dove/            DOVE：infer.py（按镜头、显存可控的推理）和微调计划
third_party/              上游仓库，以 git submodule 引入，不做修改：DOVE（训练/推理）、DOVER（视频质量指标）；
                          权重放在各自目录内且不入库（DOVE/pretrained_models/、DOVER/pretrained_weights/DOVER.pth）
docs/                     ugc_dataset.md（数据集流程与调研，先读这个）、ugc_degradations.md（真实退化与校准）、
                          hq_sources.md（高质量素材来源）
requirements/             collect.txt（.venv）、ml.txt（.venv-iqa）、dove.txt（.venv-dove）
data/        （不入库）   real_ads/  hq/  stats/  pairs/  benchmarks/  assets/，详见 data/README.md
outputs/     （不入库）   calibration/（合成数据的指标、校准分组）  runs/（模型输出）  figures/  logs/
```

## 初始化

```bash
git clone --recurse-submodules <this repo>      # 已有的克隆：git submodule update --init
.venv-iqa/bin/python -m adup.ugc.fonts          # 下载字幕字体到 data/assets/fonts/
```

## 环境

| venv | Python | 用途 |
|---|---|---|
| `.venv` | 3.11 | 广告采集和界面截图（selenium）。见 `requirements/collect.txt`。 |
| `.venv-iqa` | 3.11 | 素材下载、测量、生成配对（torch 2.11、transformers 5）。见 `requirements/ml.txt`。 |
| `.venv-dove` | 3.11 | DOVE 推理和训练（torch 2.8 cu128、diffusers 0.33、transformers 4.51），和 `.venv-iqa` 隔开以免版本冲突。见 `requirements/dove.txt`。 |

## 流程

```bash
# ① 真实广告（目标 LQ 域）
.venv/bin/python -m adup.real_ads.tiktok_topads
.venv/bin/python -m adup.real_ads.meta_adlib_web
.venv/bin/python -m adup.real_ads.contact_sheets data/real_ads/<source>/<date>

# ② 测量真实广告 -> data/stats/real_ads/（管线读取的校准目标）
.venv-iqa/bin/python -m adup.analysis.ugc_look      tiktok data/stats/real_ads/ugc_look.csv <videos...>
.venv-iqa/bin/python -m adup.analysis.shots         data/real_ads/<source>/<date>/shots <videos...> --table data/stats/real_ads/shots.csv
.venv-iqa/bin/python -m adup.analysis.ugc_content   ads data/stats/real_ads/content_ads.csv <videos...>
.venv-iqa/bin/python -m adup.analysis.text_overlay  tiktok_720p data/stats/real_ads/text_overlay.csv <videos...>

# ③ 高质量素材 + 预筛 + 主题标签 -> data/hq/、data/stats/hq/
.venv-iqa/bin/python -m adup.hq.ultravideo 300 --per-source 4 --kind both              # 4K -> data/hq/ultravideo/4k/
.venv-iqa/bin/python -m adup.hq.ultravideo 1000 --per-source 10 --res 8k --kind both   # 8K -> data/hq/ultravideo/8k/
.venv/bin/python     -m adup.hq.ui_screens --n 300                                     # 录屏页面；--dpr 6 做 4K
.venv-iqa/bin/python -m adup.hq.gate --manifest data/hq/ultravideo/4k/manifest.csv --gt-short 1440 --out data/hq/ultravideo/4k/gate_1440.csv
.venv-iqa/bin/python -m adup.analysis.ugc_content clips    data/stats/hq/content_clips.csv data/hq/ultravideo/{4k,8k}/manifest.csv
.venv-iqa/bin/python -m adup.analysis.ugc_content pool     data/stats/hq/content_pool.csv
.venv-iqa/bin/python -m adup.analysis.ugc_content stills   data/stats/hq/content_unsplash.csv
.venv-iqa/bin/python -m adup.analysis.ugc_content coverage data/stats/hq/content_coverage.csv

# ④⑤ 编排 + 生成配对 -> data/pairs/<数据集>/（参数：configs/pairs/v6.yaml；--scale auto 让 LQ 短边在 720/576/540 之间抽）
.venv-iqa/bin/python -m adup.ugc.director --clips data/hq/ultravideo/{4k,8k}/manifest.csv \
    --gate data/hq/ultravideo/{4k,8k}/gate_1440.csv --stills data/stats/hq/content_unsplash.csv \
    --screens data/hq/ui_screens/manifest.csv --gt-short 1440 --n-ads 5000 --out data/pairs/ugc_v5_2k/specs.jsonl
.venv-iqa/bin/python -m adup.make_pairs --sequences data/pairs/ugc_v5_2k/specs.jsonl --gt-short 1440 --scale auto
# 不经编排、每条素材直接做一个单镜头配对（对照用）
.venv-iqa/bin/python -m adup.make_pairs --manifest data/hq/ultravideo/4k/manifest.csv --out data/pairs/plain_2k_x2 --scale 2

# 校准：给合成集打分（写到 outputs/calibration/），再和真实广告比较分布
.venv-iqa/bin/python -m adup.analysis.degradation_stats <group> outputs/calibration/degradation.csv <videos...>
.venv-iqa/bin/python -m adup.analysis.quality           <group> outputs/calibration/quality.csv     <videos...>
.venv-iqa/bin/python -m adup.analysis.compare tiktok_720p <group>

# DOVE 推理（按镜头；权重放在 third_party/DOVE/pretrained_models/DOVE）
.venv-dove/bin/python training/dove/infer.py --out outputs/runs/dove/<name> --upscale 4 <videos...>
```

本机只做测试；批量下载和生成在服务器上跑（见 `docs/ugc_dataset.md` 第 7 节）。

## 数据来源与许可

| 数据 | 位置 | 许可 / 条款 |
|---|---|---|
| TikTok Top Ads | `data/real_ads/tiktok_topads/` | 广告主的素材。只用于分析和评测。 |
| Meta 广告库 | `data/real_ads/meta_adlib_web/` | 广告主的素材。只用于分析和评测。 |
| UltraVideo | `data/hq/ultravideo/` | CC-BY-4.0 + **仅限非商业研究**（素材来自 YouTube）。 |
| Unsplash Lite | `data/hq/unsplash_lite/` | Unsplash Lite 许可，允许内部商用模型训练。 |
| KwaiVIR（NTIRE 2026） | `data/benchmarks/KwaiVIR/` | 快手竖屏短视频修复基准，研究用途；1080x1920、6 秒、HEVC。 |
| VideoLQ | `data/benchmarks/VideoLQ/` | 研究基准集。 |
| HQ-VSR（DOVE） | `data/benchmarks/HQ-VSR/` | 源自 OpenVid-1M。研究用途。 |

Pexels、Pixabay 和 Mixkit 禁止脚本批量下载，Pexels 和 Pixabay 还明确禁止用于机器学习。
商用训练需要自己拍摄、创作者授权或购买的素材。
