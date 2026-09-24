# 高质量（GT）视频来源调研

GT 需要满足五个条件：
1. **真实清晰**：在目标分辨率下确实有细节。×4 做 4K 竖屏，需要 2160x3840 竖屏素材或 8K 横屏素材；
   ×2 做 1080p 竖屏，需要 4K 横屏或 1080p 竖屏。
2. **内容像 UGC**：人物口播、手持产品、美食、宠物、穿搭、美妆，手机拍摄质感。
3. **最好原生竖屏**：横屏裁竖屏会损失约 2/3 的像素，而且构图不对。
4. **压缩少**：原始或高码率文件。
5. **许可证**：区分只能研究用和可以商用。

## 0. 当前规则（2026-09-24）

- **GT 至少 2K**：短边 ≥ 1440，只缩小不放大（`--gt-short 1440`；4K 用 2160）。
- **横竖都要**：画幅按 9:16 45%、16:9 35%、4:5 10%、1:1 10% 抽取。
  如果某个画幅需要放大才能凑出，就不用这个画幅。
- **按画幅看素材要求**：
  | 目标 GT | 4K 横屏素材 | 8K 横屏素材 |
  |---|---|---|
  | 16:9 横屏 2560x1440 / 3840x2160 | 可以 | 可以 |
  | 4:5、1:1（短边 1440） | 可以 | 可以 |
  | 9:16 竖屏 1440x2560 | 不行，4K 最多裁出 1215x2160 | 可以（裁 2430x4320 再缩小） |
  | 9:16 竖屏 2160x3840 | 不行 | 可以 |

  所以 **2K 以上的 9:16 竖屏 GT 只能来自 8K 素材或原生竖屏 4K 素材**。
- **有效分辨率门槛**：在 GT 分辨率下，取最清晰的区域做 ×2 下采样再上采样，PSNR 必须 ≤ 38 dB（`adup/sources/gate.py`）。
  否则说明这段"4K"素材在 GT 尺寸下没有真实细节。在 2K 横屏 GT 上，UltraVideo 4K 大约 60% 能通过；
  直接当 4K GT 用时，只有约 30% 通过。

## 1. 现状（2026-09）

我们目前只用了 UltraVideo 的一部分，每个源视频取 1 条，每类最多 25 条，共 135 条。按现有的人物筛选规则统计：

| | 4K（3840 宽） | 8K（7680 宽） |
|---|---|---|
| 符合筛选的片段 | 8,165 条，来自 432 个源视频 | 229 条，来自 42 个源视频 |
| 美食 | 2,934 | 167 |
| 手持产品 | 2,194 | 39 |
| 口播 | 1,604 | 7 |
| 居家生活 | 704 | 6 |
| 穿搭 | 421 | 7 |
| 宠物 | 299 | 3 |
| **美妆** | **9** | **0** |
| 竖屏片段 | 0 | 0 |

结论：
- **×2（540p → 1080p）**：UltraVideo 4K 还能扩大约 10 倍，从 135 条扩到每个源取多条的上千条。
- **×4（540p → 4K）**：UltraVideo 8K 只有 229 条，其中 73% 是美食，不够用。
- **美妆几乎为零**，所有素材都是横屏电影感。这两个缺口靠公开数据集补不上。

## 2. 研究用数据集（不能商用）

| 数据集 | 规模 / 分辨率 | 内容 | 许可 / 获取 | 适合做什么 |
|---|---|---|---|---|
| UltraVideo（在用） | 4.2 万条，4K，22% 为 8K | YouTube 4K/8K，百余类主题 | CC-BY-4.0 + 非商业研究；HF 直接下载 | ×2 主力；8K 做少量 ×4 |
| UltraVideo-Long | 长片段版 | 同上 | 同上 | 需要更长连续镜头时用 |
| OpenHumanVid（CVPR 2025） | 5230 万条人物片段，52.7% 高于 1080p | 电影、电视剧、纪录片 | 填表申请，研究用 | 人物多，但以横屏电影为主，大多到不了 4K |
| OpenVidHD-0.4M | 43 万条 1080p | OpenVid-1M 的高清子集（DOVE 的 HQ-VSR 就来自 OpenVid） | CC-BY-4.0，但需遵守原素材许可 | 1080p 横屏裁竖屏只剩 608x1080，不够做 1080p 竖屏 GT |
| HumanVid | 2 万条 1080p 人物视频，含竖屏 | 来自 Pexels | Pexels 条款禁止用于机器学习 | 不建议 |
| Koala-36M / MiraData / VidGen-1M | 百万级，多数 ≤1080p | YouTube | 研究用 | 分辨率不够做 GT |
| KwaiVIR（NTIRE 2026，已下载到 `data/public/KwaiVIR`） | 训练集：200 条合成 HQ/LQ + 48 条真实；验证 11 条；测试 20+ 条。全部 1080x1920 竖屏、30fps、180 帧（6 秒）、HEVC，HQ 码率 17–36 Mbps | 快手竖屏短视频 | Codabench 公开链接，研究用途 | **领域最接近**。但它是同尺寸修复（LQ 和 HQ 一样大），不是超分；1080p 也低于 2K，所以只做评测和领域参考，不当 GT |
| RealVSR / MVSR4x / RealMCVSR | 几百对 | 手机多摄像头同时拍摄的真实 LR-HR 配对 | 研究用 | 评测真实手机退化，数据量小 |
| UVG / BVI-DVC / Inter4K | 几十到上千条 4K 序列 | 专业拍摄的测试序列 | 多为非商业 | 干净的 4K 评测集，不像 UGC |

## 3. 可以商用的来源

| 来源 | 说明 |
|---|---|
| **委托 UGC 创作者拍摄** | 通过 Billo、Insense、Upwork 等平台找创作者，按拍摄清单用新款手机拍竖屏 4K，并在合同里写明允许用于 AI 训练。**领域最匹配**（真实手机、真实 UGC、可补美妆），也能拿到高码率原片。按条计价，需要询价。 |
| **自己拍** | 2–3 台近两年的 iPhone 或安卓旗舰，竖屏 4K30/60，最高码率（iPhone 可用 ProRes 或 Apple Log），按品类拍摄清单拍。成本最低、可控性最好。 |
| **AI 训练数据授权市场** | Troveo 称有 800 万小时已获得 AI 训练授权的视频；Shutterstock 已有向大模型公司出售数据授权的业务。适合大规模，但需要询价和法务审核。 |
| **Wikimedia Commons** | 约 4,400 个 3840x2160 视频，许可为 CC-BY / CC-BY-SA / CC0，允许商用但要署名，CC-BY-SA 需要法务确认。内容以自然、航拍为主，人物很少。 |
| **Vimeo 知识共享视频** | NIST 的 V3C1/V3C2 共约 1.7 万条 CC 视频（研究集）；也可以直接找 Vimeo 上允许下载原片的 CC-BY 视频。质量参差，需要过滤。 |

Pexels、Pixabay、Mixkit 的条款禁止脚本批量下载，Pexels 和 Pixabay 还明确禁止用于机器学习，所以不在候选里。

## 4. 建议

1. **现在（研究阶段）**：
   - 把 UltraVideo 4K 扩到每个源取 3–5 条，得到约 1,500 条片段，用于 ×2。
   - 取全部 8K 人物片段（229 条），先跑通 ×4 流程。
   - KwaiVIR 已下载，作为竖屏短视频领域的评测集。
2. **补美妆和竖屏**：公开数据补不了，需要自己拍或委托创作者拍。建议先拍 200–500 条竖屏 4K，
   重点是美妆护肤、手持产品、口播、美食，同时解决商用许可问题。
3. **商用模型**：训练集只能用第 3 节的来源。UltraVideo、OpenHumanVid 等研究数据只能用于实验和论文。

## 参考
- KwaiVIR 数据下载（Codabench 研究版页面上的 Google Drive 链接）：https://www.codabench.org/competitions/15280/
- UltraVideo：https://huggingface.co/datasets/APRIL-AIGC/UltraVideo ，论文 https://arxiv.org/abs/2506.13691
- OpenHumanVid：https://github.com/fudan-generative-vision/OpenHumanVid
- OpenVid-1M / OpenVidHD：https://huggingface.co/datasets/nkp37/OpenVid-1M
- HumanVid：https://humanvid.github.io/
- KwaiVIR（NTIRE 2026）：https://arxiv.org/abs/2604.10551 ；KwaiSR（NTIRE 2025）：https://arxiv.org/abs/2504.15003
- RealVSR：https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Real-World_Video_Super-Resolution_A_Benchmark_Dataset_and_a_Decomposition_Based_ICCV_2021_paper.pdf
- MVSR4x：https://arxiv.org/abs/2212.05342
- V3C（Vimeo CC）：https://catalog.data.gov/dataset/vimeo-creative-commons-collection-v3c2
- Wikimedia Commons 4K 视频：https://commons.wikimedia.org/wiki/Category:4K_videos
- Troveo：https://www.troveo.ai/resources/ai-training-data-marketplace
