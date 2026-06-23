# AURA Streaming — Bug Tracker & Troubleshooting

追蹤 AURA Omni **streaming video** E2E 驗證過程中遇到的問題、根因與修復狀態。  
環境參考：branch `AURA_streaming_input`，deploy `/tmp/aura_omni_gpu23.yaml`（GPU2=AURA，GPU3=ASR+TTS+Code2Wav），port **8010**。

**最後更新**：2026-06-22

---

## 狀態圖例

| 標記 | 含義 |
|------|------|
| ✅ 已修 | 程式已合入本地 / 待 commit |
| 📋 設計 | 預期行為，非 bug |
| 🔧 操作 | 無需改碼，調參或流程即可 |
| ⚠️ 待驗 | 修復已做，需重啟 server 後再跑 E2E 確認 |

---

## 1. Client / Demo

### 1.1 `0 turn(s)` — 從未觸發推理

| 項目 | 內容 |
|------|------|
| **現象** | Demo 結束顯示 `0 turn(s)`，server 無 `response.start` |
| **根因** | EVS（`enable_frame_filter`）把靜態/近重複畫面濾掉，`frame_buffer` 長期不足 `auto_trigger_min_frames` |
| **修復** | Client 加 `--no-evs`（`session.config` 設 `enable_frame_filter: false`） |
| **狀態** | ✅ 已修（`streaming_video_demo.py`） |
| **驗證** | 正式測試影片加 `--no-evs` |

### 1.2 Demo 啟動卡住 30–60s / Ctrl+C 在 import

| 項目 | 內容 |
|------|------|
| **現象** | 跑 `streaming_video_demo.py` 無輸出，Ctrl+C 堆疊在 `import vllm_omni` → `transformers` |
| **根因** | Demo 曾 `from vllm_omni.entrypoints.openai.aura_session_history import ...`，拉起整個 vLLM/transformers |
| **修復** | Demo 內聯 `is_effectively_silent` / `SILENT_TEXT`，不再 import `vllm_omni`；啟動印 `Connecting to ...` |
| **狀態** | ✅ 已修 |
| **驗證** | `python examples/.../streaming_video_demo.py --help` 應 <1s |

### 1.3 Server 未 ready 就連線

| 項目 | 內容 |
|------|------|
| **現象** | Client 連上後長時間無反應 |
| **根因** | 四 stage 冷啟動可達 **~10+ 分鐘**（log：`AsyncOmniEngine initialized in 696s`），需等 `Application startup complete` |
| **修復** | 🔧 先 `curl http://localhost:8010/health`，確認 200 再跑 demo |
| **狀態** | 📋 操作 |

---

## 2. Server / Handler

### 2.1 首輪 orchestrator 崩潰 — `tts_ref_audio` 缺失

| 項目 | 內容 |
|------|------|
| **現象** | Streaming 首請求 orchestrator / Stage-2 報錯缺少 TTS ref |
| **根因** | `AuraStreamingVideoSessionConfig` 未帶預設 `tts_ref_audio` / `tts_ref_text` |
| **修復** | `serving_video_stream.py`：`AuraStreamingVideoSessionConfig` 補預設 ref 路徑與文案 |
| **狀態** | ✅ 已修 |
| **檔案** | `vllm_omni/entrypoints/openai/serving_video_stream.py` |

### 2.2 SessionHistory 只累積 assistant、丟失 user/video

| 項目 | 內容 |
|------|------|
| **現象** | `AURA turn prompt` log 出現連續 `assistant` / `<|silent|>`，無 `user`；`videos` 只有 1 段 |
| **根因** | `asr2aura_session` 在**臨時** history 上 `add_user_message`；`on_turn_complete` **只** `add_assistant_message`，未寫回 `AuraSessionState` |
| **修復** | ① `_process_query_engine` 暫存 `pending_turn_video` ② `asr2aura_session` → `record_turn_transcript(request_id)` ③ `on_turn_complete` 先 `add_user_message(transcript, video)` 再 `add_assistant_message`（**純 video 輪 transcript 可為空**） |
| **狀態** | ✅ 已修 |
| **檔案** | `serving_video_stream.py`, `aura_omni.py` |

### 2.3 ASR 原文帶 wrapper 進 prompt

| 項目 | 內容 |
|------|------|
| **現象** | `transcript='language Chinese<asr_text>画面有什么？'` |
| **根因** | Qwen3-ASR 輸出格式未清理 |
| **修復** | `_clean_asr_transcript()`：`language ...<asr_text>` → 純文字 |
| **狀態** | ✅ 已修 |
| **檔案** | `aura_omni.py` |

### 2.4 多輪 prompt 缺少 role 後換行

| 項目 | 內容 |
|------|------|
| **現象** | `<|im_start|>systemYou are...`（無 `\n`） |
| **根因** | `get_vllm_inputs()` 與單輪 `asr2aura` 格式不一致 |
| **修復** | `get_vllm_inputs()`：`<|im_start|>{role}\n`，結尾 `assistant\n` |
| **狀態** | ✅ 已修 |
| **檔案** | `aura_session_history.py` |

### 2.5 Context history 與 silent

| 項目 | 內容 |
|------|------|
| **現象** | Prompt 裡大量 `<|silent|>`，懷疑 context history 塞錯 |
| **釐清** | **Context history（prune 後）**：Rule B — silent 輪**不寫入**（`_rewrite_qa_for_history`）✅ |
| | **Sliding window（prune 前）**：原版設計**保留** silent + video（近 N 輪）📋 |
| | **先前 log 異常**：主因是 §2.2 user 未持久化，不是 Rule B 失效 |
| **狀態** | 📋 設計 + ✅ §2.2 修復後應恢復 user/assistant 交替 |

---

## 3. Pipeline / Stage Processors

### 3.1 Stage-3 無限 polling（TTS 空 codec）

| 項目 | 內容 |
|------|------|
| **現象** | Code2Wav stage 一直等，pipeline 不結束 |
| **根因** | TTS 空 codec 時 `talker2code2wav_full_payload` 回 `None`，無 finished sentinel |
| **修復** | `qwen3_tts.py`：flush 送 empty finished sentinel；`omni_connector_model_runner_mixin.py`：`force_finished=True` + `None` 時 fallback payload |
| **狀態** | ✅ 已修 |
| **檔案** | `qwen3_tts.py`, `omni_connector_model_runner_mixin.py` |

### 3.2 第二次 run Stage-3 `AssertionError`（`num_scheduled_tokens=-1`）

| 項目 | 內容 |
|------|------|
| **現象** | 第二輪 TTS/Code2Wav `prompt_len=0` 請求崩潰 |
| **根因** | `talker2code2wav_token_only` 建立空 prompt；scheduler 算出負 token 數 |
| **修復** | `qwen3_tts.py`：`prompt_len<=0` 跳過；`gpu_generation_model_runner.py`：`total_num_scheduled_tokens<=0` 提早返回 |
| **狀態** | ✅ 已修 |
| **檔案** | `qwen3_tts.py`, `gpu_generation_model_runner.py` |

### 3.3 退化輸出 ` ﹑` 仍走 TTS

| 項目 | 內容 |
|------|------|
| **現象** | Turn 40+ 連發標點字元，TTS 空轉 |
| **根因** | 僅跳過精確 `<|silent|>`，模型常吐標點 filler |
| **修復** | `aura_text_utils.py`：`AURA_PUNCT_CHARS` + `is_punctuation_only_text()`；`is_effectively_silent()` 用於 `aura2tts`、`on_turn_complete`、history 正規化、demo 顯示 `(silent)` |
| **狀態** | ✅ 已修（體驗優化，非根因） |
| **檔案** | `aura_text_utils.py`, `aura_session_history.py`, `aura_omni.py`, `serving_video_stream.py`, `streaming_video_demo.py` |

### 3.4 Prompt 除錯 log

| 項目 | 內容 |
|------|------|
| **用途** | 每輪 Stage-1 組好 prompt 後印骨架 + video shape |
| **用法** | Server log 搜 `AURA turn prompt`（**AURA worker**，非 API 主進程） |
| **狀態** | ✅ 已加 |
| **檔案** | `aura_omni.py` → `_summarize_vllm_inputs()` |

---

## 4. 行為釐清（非 Bug）

| 現象 | 說明 |
|------|------|
| 首請求極慢 | 四 stage JIT / CUDA graph 冷啟動；🔧 先暖身（`example.mp4` + `--text-only --max-duration 3`） |
| 影片內嵌音軌無效 | Demo 用 OpenCV 只送幀；語音靠 `audio.chunk` / `--audio-schedule` |
| Turn 2 承諾「幫你盯泳池」 | 對 `02_pool_notify.wav` 的正常回應，非 SessionHistory 組錯 |
| ` ﹑` vs `<|silent|>` | 前者是模型生成的標點字元；後者是 special token（§3.3 已當 silent 處理） |
| `NVFP4 W4A4 weight_scale NaN-clamp: installed` | `vllm_omni/patch.py` import 時一行的確認 log，**不是卡住**；AURA bf16 路徑通常不執行 clamp |
| 重啟 server 需重新暖身 | Worker 不熱載入 |

---

## 5. 建議 E2E 命令

### 暖身（每次重啟 server 後）

```bash
/public/wtk/.venv/bin/python examples/online_serving/aura_omni/streaming_video_demo.py \
  --url ws://localhost:8010/v1/video/chat/stream \
  --model aurateam/AURA \
  --video /public/wtk/AURA/AURA/AURA_bench_eval/StreamingBench/figs/example.mp4 \
  --burst-interval 0 --fps 2 --text-only --max-duration 3 --no-evs
```

### 正式測試

```bash
/public/wtk/.venv/bin/python examples/online_serving/aura_omni/streaming_video_demo.py \
  --url ws://localhost:8010/v1/video/chat/stream \
  --model aurateam/AURA \
  --video /public/wtk/aura_prompts/aura_test.mp4 \
  --burst-interval 0 --fps 2 \
  --audio-schedule 1:/public/wtk/aura_prompts/01_frame_what.wav \
  --audio-schedule 10:/public/wtk/aura_prompts/02_pool_notify.wav \
  --no-evs
```

可選：`--max-duration 55`、`--auto-trigger-min-frames 4`（減少純畫面空轉）。

---

## 6. 修改檔案索引

| 檔案 | 主要變更 |
|------|----------|
| `entrypoints/openai/serving_video_stream.py` | TTS ref 預設、turn 持久化、`pending_turn_video` |
| `entrypoints/openai/aura_session_history.py` | silent 正規化、prompt `\n`、`is_effectively_silent` |
| `entrypoints/openai/aura_text_utils.py` | 共用標點判斷 |
| `entrypoints/openai/aura_cross_turn_penalty.py` | 復用 `is_punctuation_only_text` |
| `entrypoints/openai/video_stream_base.py` | `on_turn_complete(..., request_id)` |
| `model_executor/stage_input_processors/aura_omni.py` | ASR 清理、transcript 登記、prompt log |
| `model_executor/stage_input_processors/qwen3_tts.py` | 空 codec sentinel、`prompt_len<=0` |
| `worker/omni_connector_model_runner_mixin.py` | TTS finished fallback |
| `worker/gpu_generation_model_runner.py` | `num_scheduled_tokens<=0` 提早返回 |
| `examples/.../streaming_video_demo.py` | `--no-evs`、`--audio-schedule`、輕量 import |

---

## 7. 待辦 / 可選優化

- [ ] ⚠️ 重啟 server 後跑兩次正式 demo，確認 §2.2 / §3.1 / §3.2 不再復現
- [ ] 可選：sliding window 也省略純 silent 輪（減 token / 空轉）
- [ ] 可選：AURA stage `temperature` 0.5→0.0（`aura_omni_gpu23.yaml`）
- [ ] Commit 本地修復並對齊 upstream PR

---

## 相關文件

- [aura_video_stream_api.md](aura_video_stream_api.md) — WebSocket 協議與 session 參數
- [../user_guide/examples/online_serving/aura_omni.md](../user_guide/examples/online_serving/aura_omni.md) — 範例說明
