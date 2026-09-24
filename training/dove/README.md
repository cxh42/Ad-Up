# 针对 UGC 广告微调 DOVE

上游 DOVE 代码原样保留在 `third_party/DOVE`。本项目特有的内容都放在这个目录：启动脚本、DeepSpeed/accelerate
配置，以及数据集加载器的子类。

## 影响我们数据的上游事实
- **Stage 1** 在 `25x320x640` 的片段上训练（帧数 x 高 x 宽，帧数必须是 8N+1）。每个样本先读连续 35 帧，
  再从中裁出 25 帧。
- **Stage 2** 在 `2x320x640` 的片段上训练，`--image_ratio 0.8`，即 80% 的样本是单张图片。
- **LQ 输入**只从 HQ 在线合成，使用 `finetune/configs/degradation*.yaml`（RealBasicVSR 风格的退化流程），倍率 ×4。
- **推理**接受任意宽高（补齐到 16 的倍数），支持空间分块（`--tile_size_hw`）和时间分段（`--chunk_len`）。
  `--upscale` 必须是整数。

## 计划的改动
- **竖屏裁剪**：在横屏 `320x640` 之外，增加 `train_resolution 17x640x320`。
- **按镜头切分数据**：在转场处切开片段。≥27 帧的镜头进视频池，更短的镜头进 stage 2 用的图片池。
- **配对数据**：从 `data/pairs/<set>/<clip>/{gt.mp4, lq_*.mp4}` 读取离线生成的配对，替代在线退化，或两种样本混用。
  注意三点：
  - GT 可能是 9:16、4:5 或 1:1，尺寸见 `meta.json` 的 `gt_size`、`lq_size`。
  - VP9/AV1 的 LQ 已解码存成无损 H.264（`stored_as: h264_lossless`），decord 可以直接读。
  - `meta.json` 的 `text` 里有每个文字元素的帧范围和位置，可以用来做文字区域的损失加权或单独评测。
- **评测**：在真实广告（`data/real_ads/...`，无参考）、`data/benchmarks/VideoLQ` 和 `data/benchmarks/KwaiVIR` 上评测。

预训练权重按上游 README 放到 `third_party/DOVE/pretrained_models/`。训练输出写到 `outputs/runs/dove/<run_name>/`。
