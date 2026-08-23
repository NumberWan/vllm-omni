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

## AURA 快速開始

本分支（`AURA_production_v026`）提供 AURA 即時音訊／視訊 pipeline：
`Qwen3-ASR → AURA_v2 → Qwen3-TTS → Code2Wav`。

全新環境先安裝：

```bash
bash scripts/install_aura_omni.sh
```

### 只需 API／WebSocket 接口

兩卡生產向 helper（預設 `:8666`、`aura_omni_2gpu_best`）：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/start_aura_omni.sh
curl http://127.0.0.1:8666/v1/models
```

本地 AURA_v2 權重：

```bash
MODEL=/workspace/models/AURA_v2 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/start_aura_omni.sh
```

亦可用 `vllm serve /workspace/models/AURA_v2 --omni`（預設埠 `8000`；固定埠加
`--port 8666`）。WebSocket：`/v1/video/chat/stream`。

開 safe 工具時：

```bash
export VLLM_AURA_TOOL_EXECUTOR=safe
# 可選 API key（唔好寫入檔／git；改 key 後要重啟 server）
export DEEPSEEK_API_KEY='sk-...'   # 無 → DeepSeek mock
export SERPER_API_KEY='...'        # 無 → WebSearch 唔係真搜
# 可選（預設關）：VLLM_AURA_TOOL_BRAVE / _DDG / _WEBFETCH
CUDA_VISIBLE_DEVICES=0,1 bash scripts/start_aura_omni.sh
```

計算機、上海時間、天氣、匯率、IP 定位唔使 key。`start_aura_omni.sh` 會把上列
env 傳入 server 進程。Session 要設 `"tool_mode": "auto"`（Web demo 腳本已設）。

### 需要瀏覽器 Web Demo（正式：Native 外觀）

一條指令啟動 AURA_v2（1-GPU）+ 原版 Native 前端 bridge（自動揀 <2 GiB 空卡，
唔殺其他人進程；指定卡：`AURA_GPU=N`）：

```bash
# 同上，可選先 export DEEPSEEK_API_KEY / SERPER_API_KEY
bash examples/online_serving/aura_omni/native_gateway_web_demo/run_1gpu_stack.sh
```

開啟：

- 本地：`http://127.0.0.1:9999/`
- LAN：`http://<host-ip>:9999/`
- AURA backend：`:8666`

停止：

```bash
bash examples/online_serving/aura_omni/native_gateway_web_demo/stop_1gpu_stack.sh
```

預設：`VLLM_AURA_TOOL_EXECUTOR=safe`、`TOOL_MODE=auto`、`max_tool_depth=3`、
鏡頭 `AUTO_TRIGGER=1`（至少 2 幅可開輪）、Base voice clone、單 session。
介面支援 camera、PTT、ASR 文字、工具氣泡與串流 TTS；PTT 會停耳邊舊播放
（frontend barge-in，唔 cancel GPU 上已跑緊嘅 TTS）。未接打字發話／`/eval`。

本地已有權重時：

```bash
MODEL=/workspace/models/AURA_v2 \
bash examples/online_serving/aura_omni/native_gateway_web_demo/run_1gpu_stack.sh
```

### 另一套 UI：MiniCPM 風格（可選）

1-GPU MiniCPM demo stack：

```bash
bash examples/online_serving/aura_omni/minicpm_style_web_demo/run_1gpu_demo_stack.sh
bash examples/online_serving/aura_omni/minicpm_style_web_demo/stop_1gpu_demo_stack.sh
```

2-GPU + 兩個 demo 埠（`:7862` + `:7863`，共用同一個 AURA）：

```bash
bash examples/online_serving/aura_omni/minicpm_style_web_demo/run_2gpu_dual_demo_stack.sh
bash examples/online_serving/aura_omni/minicpm_style_web_demo/stop_2gpu_dual_demo_stack.sh
```

呢條線預設唔開 safe tools；要工具先 `export VLLM_AURA_TOOL_EXECUTOR=safe`
同上面嘅 key。

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

完整安装、WebSocket／HTTP、工具協議、離線模型及故障排查請看：

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
