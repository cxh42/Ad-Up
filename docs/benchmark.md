# UGC 广告视频超分基准（设计与实现状态）

目标：在 UGC 广告上，比较从插值、小模型到扩散模型的各类视频超分方法，同时看**画质**、**内容保真**（文字、商品、人脸不能被改）
和**成本**。通用部分按业界标准，便于和文献对照；广告特有的部分（保真指标、真实成对赛道、再转码后的质量）是这个基准的贡献点。

代码：`adup/bench/`（`run.py` 跑方法，`evaluate.py` 打分，`metrics.py` 公共指标）。

## 1. 赛道

| 赛道 | 输入 → 输出 | 参考 | 状态 |
|---|---|---|---|
| A 合成主赛道 | 360 / 540 / 720p → 1080p（1080×1920 / 1920×1080） | GT | 已实现 |
| B ×4 兼容子集 | 270p → 1080p | GT | 已实现（网格里的 270） |
| C 真实赛道 | 真实广告 360p（Meta SD）/ 576–720p（Meta HD、TikTok）→ 1080p | 无 | 已实现（`evaluate_real.py`：DOVER、MUSIQ、CLIP-IQA） |
| D 真实成对赛道 | Meta 360p → 与同一条广告的 Meta 720p 比较 | 近似参考（同源、质量更高，但也压缩过） | 已实现（`evaluate_real.py`：输出缩到 720p 后算 PSNR / SSIM / LPIPS） |

## 2. 测试集

- **合成 dev / test**：按源视频划分（`ugc.director.split`，合成前分好），`director --only-split dev|test` 编排，
  `make_pairs --grid` 生成：每条 GT 一个 LQ 对应 尺寸 {270, 360, 540, 720} × 编码 {H.264, VP9}，退化用 calibrated（按真实广告校准）。
  重退化子集另跑：`--grid --set degrade.grid.rbvsr_prob=1.0`（RealBasicVSR 原版范围，和训练集同分布）。
  建议规模：dev 50 条、test 100 条广告（各 8 个 LQ）。
- **真实**：`data/stats/real_ads/splits.csv` 的 test 份（按广告 id 固定划分）；列表在 `data/stats/real_ads/groups/test_{hd,sd}.txt`。
- 每个 LQ 的因素都在 meta.json：尺寸 / 倍数、编码、退化强度（preset）、画幅、主题、镜头数、文字元素，结果可以按任意因素拆开。

## 3. 统一规则（`adup/bench/run.py`）

- 输出一律 1080p（`--out-short 1080`），与输入大小无关。
- 固定倍数的模型按原生倍数跑，不够就重复，超过就 area 缩到 1080p；**不先缩小输入**。
- 在输出尺寸上修复的模型（DOVE 等扩散模型）先把输入插值到输出尺寸。
- 视频模型按镜头处理：合成集用 meta.json 的精确边界，真实视频用检测到的切换（DOVE 的推理脚本自带）。
- 成本：每帧耗时（不含模型加载，模型自己报的推理时间优先）、峰值显存、机器（GPU 型号）写进每个输出旁边的 json。
- 本机测试可以用 `--max-frames N` 只跑每条视频的前 N 帧（真实广告很长）。

## 4. 指标（`adup/bench/evaluate.py`）

| 类别 | 指标 | 说明 |
|---|---|---|
| 画质（有参考） | 亮度 PSNR / SSIM（每帧）、LPIPS / DISTS（每 4 帧，pyiqa） | 业界标准 |
| 文字区域画质 | mask 内的 PSNR | mask.mkv 是合成器画的文字的逐帧 alpha |
| 文字保真 | 每个文字片段在中间帧 OCR（EasyOCR），和合成器写的原文比较：相对 GT 多出来的字符错误率（text_dcer），以及三类结果的比例 | 见下 |
| 时间稳定 | 镜头内相邻帧差与 GT 相邻帧差之差的平均（tdiff，越小越不闪） | |
| 镜头切换 | 切换前后 4 帧的 PSNR 与其余帧对比 | |
| 成本 | 每帧耗时、峰值显存 | |

**文字保真的三类结果**。OCR 在干净文字上也会出错（花体、描边、高亮字幕上字符错误率约 0.1），所以一切都相对"OCR 读 GT"的结果：
GT 上错误率超过 0.3 的片段不统计；text_dcer = 输出的字符错误率 − GT 的字符错误率，就是方法造成的文字损失。每个片段归为：
- **correct**：输出读得不比 GT 差；
- **hallucinated**：更差，但读得很确定（平均置信度 ≥ 0.8）且和 GT 读出来的不同，即"清楚的错字"，比如价格的某一位数字被改了。
  这是广告最怕的错误，普通画质指标看不出来；
- **illegible**：更差且读不确定：模糊、断裂。

OCR 以后可以换更强的（如 PaddleOCR），减少它本身的误差。

**待加**：
- 人脸区域画质和身份相似度（ArcFace 类模型；注意 insightface 模型的许可只限非商用）；
- 商品 / logo 区域的特征相似度（先用检测模型框出，子集人工核对）；
- COVER（导师组的无参考指标）；
- 再转码后的质量：输出按典型码率重编码（H.264 / AV1，1080p 与 720p 两档）后再打分，以及同画质下的码率节省；
- 帧间扭曲误差（光流）。

## 5. 人评（计划）

- 真实 test 份挑约 50 段，5–6 个方法，两两比较（2AFC），Bradley-Terry 汇总成排名；
- 分开问：画面哪个更好；文字、价格、logo 有没有看不清或被改；人脸是否自然；
- 手机尺寸竖屏全屏播放，贴近用户实际观看；
- 人评结果同时用来检验自动指标（SRCC / PLCC）。

## 6. 方法

| 类别 | 方法 | 状态 |
|---|---|---|
| 插值（下限） | bicubic、lanczos | 已接入 |
| 图片模型逐帧 | Real-ESRGAN（x4plus，spandrel 加载） | 已接入 |
| 一步扩散 | DOVE（`training/dove/infer.py --out-short 1080`） | 已接入 |
| 传统视频模型 | BasicVSR++、RealBasicVSR、RVRT | 待接入（服务器上配 mmagic 环境） |
| Meta 在用的小模型 | iVSR 的 EDSR、TSENet、Enhanced BasicVSR（OpenVINO，CPU） | 待接入 |
| 多步扩散 | Upscale-A-Video、STAR、MGLD-VSR | 待接入 |
| 一步扩散 | SeedVR2、FlashVSR | 待接入（1080p 需要大显存，服务器） |
| 我们的 | 微调后的 DOVE、实验室的模型 | 训练后 |

## 7. 运行

```bash
# 合成 dev 集：编排 + 网格生成
.venv-iqa/bin/python -m adup.ugc.director --clips data/hq/ultravideo/{4k,8k}/manifest.csv data/hq/kwaivir/manifest.csv \
    --gate data/hq/ultravideo/{4k,8k}/gate_1080.csv data/hq/kwaivir/gate_1080.csv --stills data/stats/hq/content_unsplash.csv \
    --screens data/hq/ui_screens/manifest.csv --only-split dev --n-ads 50 --out data/pairs/ugc_v7_dev/specs.jsonl
.venv-iqa/bin/python -m adup.make_pairs --sequences data/pairs/ugc_v7_dev/specs.jsonl --grid
# 跑方法（每个方法一次）
.venv-iqa/bin/python -m adup.bench.run --method dove --pairs data/pairs/ugc_v7_dev --out outputs/bench/ugc_v7_dev
# 打分
.venv-iqa/bin/python -m adup.bench.evaluate --pairs data/pairs/ugc_v7_dev --runs outputs/bench/ugc_v7_dev \
    --methods bicubic lanczos realesrgan dove --csv outputs/bench/ugc_v7_dev/results.csv
# 真实视频：跑方法，再做无参考打分和 360p→720p 半配对比较
.venv-iqa/bin/python -m adup.bench.run --method dove --list data/stats/real_ads/groups/test_sd.txt --out outputs/bench/real_test
.venv-iqa/bin/python -m adup.bench.evaluate_real --runs outputs/bench/real_test --methods bicubic dove --split test \
    --csv outputs/bench/real_test/results.csv
```
