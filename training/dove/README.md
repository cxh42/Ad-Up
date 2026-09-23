# DOVE fine-tuning for UGC ads

The upstream DOVE code is kept unmodified in `third_party/DOVE`. Everything specific to this project goes in this
folder: launch scripts, DeepSpeed/accelerate configs, and any dataset-loader subclasses.

## Upstream facts that affect our data
- **Stage 1** trains on `25x320x640` clips (frames x H x W, frames must be 8N+1). Each sample reads 35 consecutive
  frames, then crops 25 of them.
- **Stage 2** trains on `2x320x640` clips with `--image_ratio 0.8`, so 80% of the samples are single images.
- **LQ inputs** are synthesized on the fly from HQ only, using `finetune/configs/degradation*.yaml`
  (the RealBasicVSR-style pipeline), at ×4.
- **Inference** accepts any H and W (padded to multiples of 16) and supports spatial tiling
  (`--tile_size_hw`) and temporal chunking (`--chunk_len`). `--upscale` must be an integer.

## Planned changes
- **Portrait crops.** Use `train_resolution 17x640x320` alongside the landscape `320x640` crops.
- **Shot-aware data.** Split clips at scene cuts. Shots of ≥27 frames go into the video pool; shorter shots go
  into the image pool used by stage 2.
- **Paired data.** Read our offline pairs from `data/pairs/<set>/<clip>/{gt.mp4, lq_*.mp4}` instead of degrading
  on the fly, or mix both kinds of samples.
- **Evaluation.** Evaluate on real ads (`data/ads/...`, no reference) and on `data/public/VideoLQ`.

Put pretrained weights in `third_party/DOVE/pretrained_models/`, following upstream's README. Write run outputs
to `outputs/runs/dove/<run_name>/`.
