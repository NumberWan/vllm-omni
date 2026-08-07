# AURA Agent Onboarding（vLLM-Omni + Native）

給新開 Cursor Agent 的最短上手路徑。先讀本檔，再按任務深潛。

## 兩個 codebase（唔好撈亂）

| | **vLLM-Omni（production serve）** | **Native（官方 demo / 對照）** |
|---|---|---|
| 根目錄 | `/home/wtk/v0.23.0`（worktree；同 inode 常有 `/workspace/wtk/v0.23.0`） | `/home/wtk/AURA/AURA` |
| 入口 | `vllm serve … --omni` / `scripts/start_aura_omni.sh` | `start_all.sh`、README Demo Deployment |
| Pipeline | Stage0 ASR → Stage1 AURA → Stage2 Talker → Stage3 Code2Wav | 分開 ASR / VL / TTS 服務 + context manage |
| Deploy | `vllm_omni/deploy/aura_omni_2gpu_best.yaml`（預設 2 卡） | Native 腳本內 GPU 分配 |
| 1-GPU Web Demo | `examples/online_serving/aura_omni/minicpm_style_web_demo/` | — |
| 模型權重（本機） | `/workspace/models/AURA`、`Qwen3-ASR-1.7B`、hub TTS CustomVoice snapshot | 通常同 HF / 本地 cache |

## 必讀（按順序）

1. Omni serve：`examples/online_serving/aura_omni/README.md`
2. Tunables：`docs/aura/AURA_OMNI_TUNABLES.md`
3. Web demo：`examples/online_serving/aura_omni/minicpm_style_web_demo/README.md`
4. Streaming API：`docs/serving/aura_video_stream_api.md`
5. Native：`/home/wtk/AURA/AURA/README.md`（尤其 Demo Deployment、context management）

## 一鍵命令（Omni）

```bash
# API-only（預設用 bundled aura_omni.yaml / 或 start script 用 2gpu_best）
cd /home/wtk/v0.23.0
vllm serve aurateam/AURA --omni   # 或本地 /workspace/models/AURA

# 2-GPU + 兩個 Web Demo（同 backend）
bash examples/online_serving/aura_omni/minicpm_style_web_demo/run_2gpu_dual_demo_stack.sh
# stop:
bash examples/online_serving/aura_omni/minicpm_style_web_demo/stop_2gpu_dual_demo_stack.sh

# 1-GPU demo（測穩 TTS 起點；單併發）
bash examples/online_serving/aura_omni/minicpm_style_web_demo/run_1gpu_demo_stack.sh
```

Demo：`:7862` / `:7863`；AURA：`:8666`。LAN：`http://192.168.0.180:7862/`。

## 架構要點（Omni）

- YAML `devices: "0"|"1"` **相對** `CUDA_VISIBLE_DEVICES`。
- Stage1 stop：**`151669=<|silent|>`、`151645=<|im_end|>`**——冇就 silent 會 pad 到 `max_tokens`（~1.6s／turn）。
- TTS 測穩 knobs（對齊 1-GPU）：`initial_codec_chunk_frames=8`、`codec_chunk_frames=20`、Talker `T≈0.5`、`min_tokens=8`、Stage2 要有 `input_connectors.from_stage_1`。
- `prompt_len` 要用真 BPE（`VLLM_AURA_TTS_TOKENIZER`），唔好用 char heuristic。
- Web dump OK 但網頁怪：先查 client playback（預設 speed 應 1.0），唔好先改 TTS。
- End session = **新 server session**（無歷史）；UI 氣泡可能仲留住。

## OmniInteract bench

量測前同一 server 先跑完 **`0001` warmup**；報表／latency **排除 0001**。見 Cursor rule `aura-omniinteract-bench-warmup`。

## Native 對照時

- Native：片內 VAD／自家 streaming；Omni：annotation wav @ `question_time`（P6）等路徑唔同。
- 對齊 ASR／spoken 時，分清「模型輸出」vs「web playback／session flood」。
- Native 預設 stop list 同樣係 `[151669, 151645]`（eval／profile 腳本）。

## 近期相關 transcript（可引用）

Agent transcripts 喺 `/home/wtk/.cursor/projects/home-wtk/agent-transcripts/`；AURA TTS／web／2-GPU 長線討論見近期 uuid（例如 `152c1398-70ef-45d4-b844-e5bf5d389a12`）。

## 新 Agent 第一句建議任務模板

複製到新 chat：

```text
請先讀 /home/wtk/v0.23.0/docs/aura/AURA_AGENT_ONBOARDING.md，
再按任務讀 Omni README + TUNABLES，以及 /home/wtk/AURA/AURA/README.md。
工作區：Omni=/home/wtk/v0.23.0；Native=/home/wtk/AURA/AURA。
遵守 Karpathy guidelines；回覆用繁中。
任務：<在此寫你的具體問題>
```
