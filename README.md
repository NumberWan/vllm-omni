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

## AURA Web Demo 快速開始

本分支預設以瀏覽器 Web Demo 使用 AURA 即時音訊／視訊 pipeline：
`Qwen3-ASR → AURA → Qwen3-TTS → Code2Wav`。

全新環境先安裝，之後用一條指令同時啟動 AURA server、warmup 及
瀏覽器 UI：

```bash
bash scripts/install_aura_omni.sh
bash examples/online_serving/aura_omni/minicpm_style_web_demo/run_1gpu_demo_stack.sh
```

終端顯示 `Demo ready` 後開啟當中 URL。停止 1-GPU Web Demo server：

```bash
bash examples/online_serving/aura_omni/minicpm_style_web_demo/stop_1gpu_demo_stack.sh
```

如果本地沒有模型權重，首次啟動會自動從 Hugging Face 下載三個公開模型，
合計約 **26 GB**（ASR 約 4.4 GB、AURA 約 17 GB、TTS／Code2Wav 約
4.3 GB）；請另外預留安裝依賴和 cache 空間。

如果本地已有 AURA 模型權重，可直接指定路徑：

```bash
MODEL=/workspace/models/AURA \
bash examples/online_serving/aura_omni/minicpm_style_web_demo/run_1gpu_demo_stack.sh
```

腳本會等待模型載入、執行一次內建 warmup，然後印出可開啟的 URL。
warmup 使用 repo 已有的短語音及程式生成的影像，**毋須下載
OmniInteract dataset**。介面會持續串流 camera frame，支援 proactive
vision、push-to-talk、ASR 使用者文字、模型回覆及串流語音播放；目前是
放開按鈕後送出整段語音，並非 full-duplex barge-in。

### 執行 OmniInteract Smoke3 benchmark（可選）

互動試用不需要 Smoke3。只有要量度準確度、TTFT／TPOT 或做 regression
comparison 時，才需下載 OmniInteract 並執行：

```bash
bash benchmarks/omniinteract/run_streaming_bench.sh
```

benchmark 會先以 `0001` 完成整條 streaming session 作 warmup，再測
`0002`、`0003`、`0004`；報表的準確率與 latency 統計排除 `0001`。
結果寫入 `./omniinteract_bench/`，亦可用
`DATASET_PATH=/path/to/OmniInteract` 指定現有 dataset。

完整安装、WebSocket／HTTP 用法、离线模型及故障排查请看
[`examples/online_serving/aura_omni/README.md`](examples/online_serving/aura_omni/README.md)。

*Latest News* 🔥
- [2026/06] We released [0.22.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.22.0) - an **omnimodal world-model** release aligned with vLLM 0.22, featuring [Nvidia Cosmos3](recipes/cosmos3/Cosmos3-Nano.md)/DreamZero world model support, expanded quantization coverage across Blackwell/NPU/XPU, TTS production improvements, new models including MiniCPM-o 4.5, MOSS-TTS, and Lance, plus RL integration with [VeRL-Omni](https://github.com/verl-project/verl-omni).
- [2026/05] We released [0.20.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.20.0) - refreshes the serving/runtime stack for large-scale omni workloads, and improves diffusion model performance, quantization, and hardware readiness across CUDA, ROCm, MUSA, NPU, and XPU backends.
- [2026/03] We released [0.18.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.18.0) - strengthens the core runtime through a large entrypoint refactor and scheduler/runtime cleanups, expands unified quantization and diffusion execution, broadens multimodal model coverage, and improves production readiness across audio, omni, image, video, RL, and multi-platform deployments.
- [2026/03] Check out our first public [project deepdive](https://youtu.be/sgwNfsNnR9I) at the vLLM Hong Kong Meetup!
- [2026/03] **[vllm-omni-skills](https://github.com/hsliuustc0106/vllm-omni-skills)** is a community-driven collection of AI assistant skills that help developers work with vLLM-Omni more effectively. These skills can be used with popular agentic AI coding assistants like **Cursor IDE**, **Claude**, **Codex**, and more.
- [2026/02] We released [0.16.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.16.0) - A major alignment + capability release that rebases onto **upstream vLLM v0.16.0** and significantly expands performance, distributed execution, and production readiness across **Qwen3-Omni / Qwen3-TTS**, **Bagel**, **MiMo-Audio**, **GLM-Image** and the **Diffusion (DiT) image/video stack**—while also improving platform coverage (CUDA / ROCm / NPU / XPU), CI quality, and documentation.
- [2026/02] We released [0.14.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.14.0) - This is the first **stable release** of vLLM-Omni that expands Omni’s diffusion / image-video generation and audio / TTS stack, improves distributed execution and memory efficiency, and broadens platform/backend coverage (GPU/ROCm/NPU/XPU). It also brings meaningful upgrades to serving APIs, profiling & benchmarking, and overall stability. Please check our latest [paper](https://arxiv.org/abs/2602.02204) for architecture design and performance results.
- [2025/11] vLLM community officially released [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) in order to support omni-modality models serving.

---

## About

[vLLM](https://github.com/vllm-project/vllm) was originally designed to support large language models for text-based autoregressive generation tasks. vLLM-Omni is a framework that extends its support for omni-modality model inference and serving:

- **Omni-modality**: Text, image, video, and audio data processing
- **Non-autoregressive Architectures**: extend the AR support of vLLM to Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous outputs**: from traditional text generation to multimodal outputs

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

vLLM-Omni seamlessly supports most popular open-source models on HuggingFace, including:

- **Omni-modality models** (e.g. Qwen3-Omni, Cosmos, HunyuanImage, BAGEL)
- **TTS models** (e.g. Qwen3-TTS, VoxCPM2, Ming-Omni-TTS, CosyVoice3)
- **Diffusion models** — image, video, and audio generation (e.g. Qwen-Image, Wan2.2, FLUX)

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
