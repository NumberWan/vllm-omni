<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png">
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png" width=55%>
  </picture>
</p>
<h3 align="center">
Easy, fast, and cheap omni-modality model serving for everyone
</h3>

<p align="center">
| <a href="https://vllm-omni.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://deepwiki.com/vllm-project/vllm-omni"><b>DeepWiki</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> | <a href="docs/assets/WeChat.jpg"><b>WeChat</b></a> | <a href="https://arxiv.org/abs/2602.02204"><b>Paper</b></a> | <a href="https://docs.google.com/presentation/d/111-L8zF7A1j_YI_cR8JsblofdScdRr2f/edit?usp=sharing&ouid=110473603432222024453&rtpof=true&sd=true"><b>Slides</b></a> |
</p>


---

## AURA 快速开始

本分支（`AURA_production_v026`）提供 AURA 实时音视频 pipeline：
`Qwen3-ASR → AURA_v2 → Qwen3-TTS → Code2Wav`。

### 1. 安装（首次）

```bash
bash scripts/install_aura_omni.sh
```

### 2. 配置环境变量（启动前，同一 shell）

将下列变量按需替换为本地路径与密钥。**勿写入文件或提交 git**；修改后须重启服务。

```bash
# 模型路径（本地已有权重时）
export MODEL=/workspace/models/AURA_v2

# 可选 GPU：指定物理卡编号；不设则自动选择显存占用 <2 GiB 的空闲卡
# export AURA_GPU=3

# 可选工具 API（计算器 / 上海时间 / 天气 / 汇率 / IP 定位无需密钥）
export DEEPSEEK_API_KEY='sk-...'   # 未设置则 DeepSeek 使用 mock
export SERPER_API_KEY='...'        # 未设置则 WebSearch 无法真实检索
```

Web Demo 启动脚本已默认启用 `VLLM_AURA_TOOL_EXECUTOR=safe`、`TOOL_MODE=auto`、
`max_tool_depth=3`，**无需再手动 export**。

### 3. 启动浏览器 Web Demo（推荐）

```bash
bash examples/online_serving/aura_omni/native_gateway_web_demo/run_1gpu_stack.sh
```

访问：

- 本机：`http://127.0.0.1:9999/`
- 局域网：`http://<host-ip>:9999/`
- AURA 后端：`:8666`

停止：

```bash
bash examples/online_serving/aura_omni/native_gateway_web_demo/stop_1gpu_stack.sh
```

默认：Base voice clone、单 session、摄像头至少 2 帧可触发轮次。支持摄像头、PTT、
ASR 文本、工具气泡与流式 TTS；PTT 可打断当前播放（前端停止播放，不取消 GPU 上已运行的 TTS）。
暂不支持文字输入与 `/eval`。

### 4. 仅 API / WebSocket（可选）

双卡生产配置（默认 `:8666`）。此路径**不会**自动打开工具，若需要 safe 工具请自行设置：

```bash
export MODEL=/workspace/models/AURA_v2
export VLLM_AURA_TOOL_EXECUTOR=safe   # 仅 API 路径需要；Web Demo 已内置
export DEEPSEEK_API_KEY='sk-...'
export SERPER_API_KEY='...'
CUDA_VISIBLE_DEVICES=0,1 bash scripts/start_aura_omni.sh
curl http://127.0.0.1:8666/v1/models
```

亦可使用 `vllm serve /workspace/models/AURA_v2 --omni`（默认端口 `8000`；固定端口加
`--port 8666`）。WebSocket 端点：`/v1/video/chat/stream`；客户端须设置
`"tool_mode": "auto"`。

### 工具调用额外延迟（相对无工具短答「好的」）

基准：无工具短答 e2e 约 **302 ms**（ASR ~91 ms + LLM ~211 ms）。调工具时用户需等待
**Pass1 生成 tool XML → 工具执行 → Pass2 生成完整答复**；下表为相对该基准的额外等待
（2-GPU AURA_v2，`tool_mode=auto`，2026-08-23 实测）。

| 工具 | 等 HTTP | 两轮 LLM（相对「好的」~211 ms） | 用户多等 |
|---|---:|---:|---:|
| calculator | 0.5 ms | ~500 ms | +0.5 s |
| datetime | 1 ms | ~1.0 s | +1.0 s |
| currency | ~1.1 s | ~1.3 s | +2.4 s |
| weather | ~1.4 s | ~1.1 s | +2.5 s |
| DeepSeek | ~4.7 s | ~9 s | +14 s |
| WebSearch | 0.5 ms（无 key） | 模型空转最长 ~13 s | 非检索慢 |

说明：计算器、时间等本地工具执行 <2 ms，额外时间主要来自多一轮 Stage1 生成。
汇率、天气的 HTTP 为 Frankfurter / Open-Meteo 往返。DeepSeek 为真实 API 延迟叠加
第二轮长生成。WebSearch 须配置 `SERPER_API_KEY` 方可真实检索；无 key 时延迟来自模型空转。

### 运行 OmniInteract Smoke3 基准（可选）

交互试用无需 Smoke3。仅在需要测量准确率、TTFT/TPOT 或做回归对比时，下载 OmniInteract 后执行：

```bash
bash benchmarks/omniinteract/run_streaming_bench.sh
```

基准测试会先以 `0001` 完成整条 streaming session 作为 warmup，再测
`0002`、`0003`、`0004`；报表的准确率与 latency 统计排除 `0001`。
结果写入 `./omniinteract_bench/`，亦可用
`DATASET_PATH=/path/to/OmniInteract` 指定已有数据集。

完整安装、WebSocket/HTTP、工具协议、离线模型及故障排查请参阅：

- [`examples/online_serving/aura_omni/README.md`](examples/online_serving/aura_omni/README.md)
- [`examples/online_serving/aura_omni/native_gateway_web_demo/README.md`](examples/online_serving/aura_omni/native_gateway_web_demo/README.md)
- [`docs/serving/aura_video_stream_api.md`](docs/serving/aura_video_stream_api.md)

*Latest News* 🔥
- [2026/08] We released [0.26.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.26.0) - aligned with the vLLM 0.26 release line, featuring [MiniMax H3](recipes/MiniMaxAI/MiniMax-H3.md) joint video/audio generation, an experimental full-duplex realtime runtime for [MiniCPM-o 4.5](recipes/OpenBMB/MiniCPM-o-4_5.md), distributed layerwise diffusion offload, and broader model, hardware, streaming, TTS, and quantization support.
- [2026/07] We released [0.24.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.24.0) - aligned with the vLLM 0.24 release line, expanding production-ready coverage across TTS, speech, diffusion, image/video generation, and robot-policy serving, with major Omni stage runtime refactoring, diffusion request-level batching, async output materialization, quantization/cache/memory improvements, and broad CUDA/ROCm/XPU/NPU support.
- [2026/06] Starting with [0.14.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.14.0), vLLM-Omni publishes a stable release aligned with every even-numbered upstream vLLM minor version. [0.16.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.16.0), [0.18.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.18.0), [0.20.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.20.0), and [0.22.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.22.0) continued this cadence, expanding omni and world-model support with [NVIDIA Cosmos3](recipes/cosmos3/Cosmos3-Nano.md) and DreamZero, adding models such as MiniCPM-o 4.5, MOSS-TTS, and Lance, and advancing TTS, diffusion, distributed execution, quantization, RL integration through [VeRL-Omni](https://github.com/verl-project/verl-omni), and CUDA/ROCm/MUSA/NPU/XPU coverage.
- [2026/03] Check out our first public [project deepdive](https://youtu.be/sgwNfsNnR9I) at the vLLM Hong Kong Meetup!
- [2025/11] vLLM community officially released [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) in order to support omni-modality models serving.

---

## About

[vLLM](https://github.com/vllm-project/vllm) was originally designed to support large language models for text-based autoregressive generation tasks. vLLM-Omni is a framework that extends its support for omni-modality model inference and serving:

- **Omni-modality**: Text, image, audio, video, and action data processing
- **Non-autoregressive Architectures**: extend the AR support of vLLM to Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous outputs**: from traditional text generation to multimodal and action outputs

<p align="center">
  <picture>
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/omni-modality-model-architecture.png" width=55%>
  </picture>
</p>

vLLM-Omni is fast with:

- State-of-the-art AR support by leveraging efficient KV cache management from vLLM
- Pipelined stage execution overlapping for high throughput performance
- Fully disaggregation based on OmniConnector and dynamic resource allocation across stages

vLLM-Omni is flexible and easy to use with:

- Heterogeneous pipeline abstraction to manage complex model workflows
- Seamless integration with popular Hugging Face models
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Full-duplex realtime serving with streaming audio input and output (experimental)

vLLM-Omni seamlessly supports most popular open-source models on HuggingFace, including:

- **Omni-modality models** (e.g. Qwen3-Omni, MiniCPM-o 4.5, Cosmos3, HunyuanImage, BAGEL)
- **TTS models** (e.g. Qwen3-TTS, VoxCPM2, Ming-Omni-TTS, CosyVoice3)
- **Diffusion models** — image, video, and audio generation (e.g. MiniMax H3, Qwen-Image, Wan2.2, FLUX)
- **Robot-policy and action models** (e.g. GR00T-N1.7, DreamZero-DROID, InternVLA-A1, Cosmos3 action policy)

## Getting Started

Visit our [documentation](https://vllm-omni.readthedocs.io/en/latest/) to learn more.

- [Installation](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)
- [Quickstart](https://vllm-omni.readthedocs.io/en/latest/getting_started/quickstart/)
- [List of Supported Models](https://vllm-omni.readthedocs.io/en/latest/models/supported_models/)
- [Deployment Recipes](https://recipes.vllm.ai) for vLLM-Omni model serving

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM-Omni](https://vllm-omni.readthedocs.io/en/latest/contributing/) for how to get involved.

## Citation

If you use vLLM-Omni for your research, please cite our [paper](https://arxiv.org/abs/2602.02204):

```bibtex
@article{yin2026vllmomni,
  title={vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models},
  author={Peiqi Yin, Jiangyun Zhu, Han Gao, Chenguang Zheng, Yongxiang Huang, Taichang Zhou, Ruirui Yang, Weizhi Liu, Weiqing Chen, Canlin Guo, Didan Deng, Zifeng Mo, Cong Wang, James Cheng, Roger Wang, Hongsheng Liu},
  journal={arXiv preprint arXiv:2602.02204},
  year={2026}
}
```

## Join the Community
Feel free to ask questions, provide feedbacks and discuss with fellow users of vLLM-Omni in `#sig-omni` slack channel at [slack.vllm.ai](https://slack.vllm.ai) or vLLM user forum at [discuss.vllm.ai](https://discuss.vllm.ai).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=vllm-project/vllm-omni&type=date&legend=top-left)](https://www.star-history.com/#vllm-project/vllm-omni&type=date&legend=top-left)

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
