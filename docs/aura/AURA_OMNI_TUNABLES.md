# AURA Omni 可調參數（Tunables）

repo: vllm-omni (`scripts/start_aura_omni.sh`, `vllm_omni/deploy/aura_omni_2gpu_best.yaml`)
updated: 2026-07-23

**預設已係最佳 2 卡 skip stack**——多數情況唔使 set 任何 feature flag。  
一鍵拉起：

```bash
bash scripts/start_aura_omni.sh
```

量測（含 `0001` warmup，統計排除 0001）：

```bash
bash # pack remasure: AURA/20_07_2026/test/run_omni_c8_mm_flashinfer_rerun.sh
```

---

## 1. 預設係咩（唔 set = 用呢啲）

| 項目 | 預設 | 說明 |
|--|--|--|
| Deploy YAML | `vllm_omni/deploy/aura_omni_2gpu_best.yaml` | FI + Talker FI + seqs4 + mem15 + chunk20 |
| GPU | `CUDA_VISIBLE_DEVICES=2,3` | Stage0/2/3→可見卡0；Stage1→可見卡1 |
| `VLLM_AURA_STAGE0_BYPASS` | **1** | silent 真 skip Stage0（SHM inject） |
| `VLLM_AURA_SILENT_STOP_AT_STAGE1` | **1** | silent 唔喚醒 Stage2/3 TTS |
| `VLLM_AURA_TTS_GATE_ON_VOICE_ASR` | **1** | voice ASR 期間延遲 TTS prewarm |
| `VLLM_AURA_SENTENCE_TTS` | **1** | Stage1 按句中途交 TTS（贏 TTFP） |
| `VLLM_AURA_AUTO_INTERRUPT` | **0** | 自動打斷（仍未穩） |
| History service | **關** | 見下方 |

預期 c=8 voice（2 卡 skip）：ASR ~136–144、AUDIO_TTFP ~350。  
詳：[`results/vllm_omni/BEST_SKIP_2CARD.md`](AURA pack: results/vllm_omni/BEST_SKIP_2CARD.md)

---

## 2. 常用覆寫（只喺有需要時 set）

### 2.1 優先 ASR ≈ Native（~89 ms @ c=8）

```bash
VLLM_AURA_STAGE0_BYPASS=0 bash scripts/start_aura_omni.sh
```

會跑齊 silent Stage0；2 卡下 ASR 最好。`SILENT_STOP` 對 bypass0 無影響，可留預設。

### 2.2 換 GPU

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/start_aura_omni.sh
```

YAML 內 `devices: '0'|'1'` 係 **相對** `CUDA_VISIBLE_DEVICES` 嘅編號。

### 2.3 4 卡分卡（ASR｜AURA｜Talker｜Code2Wav）

```bash
DEPLOY=/public/wtk/AURA/AURA/benchmark_results/aura_mode_matrix_smoke3_20260714/deploy/async_customvoice_s1_c8_mem09_mm_flashinfer_talker_fi_seqs4_4gpu_split.yaml \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/start_aura_omni.sh
```

4 卡下真 skip 同 bypass0 Stage0 公平上接近；見 [`results/vllm_omni/C8_4GPU_SPLIT_LOG.md`](AURA pack: results/vllm_omni/C8_4GPU_SPLIT_LOG.md)。

### 2.4 Port / model / log

```bash
PORT=8667 MODEL=/models/AURA LOG_DIR=/tmp/my_aura bash scripts/start_aura_omni.sh
```

---

## 3. Feature 環境變數一覽

| Env | 預設 | 作用 | 何時改 |
|--|--|--|--|
| `VLLM_AURA_STAGE0_BYPASS` | `1` | silent 硬 skip Stage0 | 要 ASR 最優 → `0` |
| `VLLM_AURA_SILENT_STOP_AT_STAGE1` | `1` | skip 時 silent 停喺 Stage1 | 關咗 2 卡 ASR 會差好多 |
| `VLLM_AURA_TTS_GATE_ON_VOICE_ASR` | `1` | voice ASR 時 defer TTS prewarm | 少數實驗可 `0` |
| `VLLM_AURA_SENTENCE_TTS` | `1` | 按句 mid-gen → TTS | 關咗 TTFP 變差 |
| `VLLM_AURA_AUTO_INTERRUPT` | `0` | 新 query 打斷舊 generation | **唔建議開**（interrupt 未齊） |
| `VLLM_AURA_HISTORY_SERVICE` | unset | 獨立 History Unix socket | 進階；配合 `HISTORY_PRIMARY` |
| `VLLM_AURA_HISTORY_PRIMARY` | unset | History 主寫 | 同上 |
| `VLLM_AURA_LOG_TURN_PROMPT` | off | debug dump turn prompt | 除錯 |
| `VLLM_AURA_SESSION_HISTORY_DIAG` | off | history 診斷 log | 除錯 |
| `VLLM_AURA_HISTORY_VIDEO_MODE` | `all` | history 影片保留策略 | 進階 |
| `VLLM_AURA_DISABLE_VIDEO_PACK` | off | 關 video pack | 實驗 |

**已移除／唔支援**：`VLLM_AURA_TTS_PREEMPT_ON_VOICE_ASR`（跨 session abort 會 hang；代碼已清）。

---

## 4. Deploy YAML 旋鈕（改檔，唔係 env）

檔案：`aura_omni_2gpu_best.yaml`（或 `DEPLOY=` 指向嘅 yaml）

| 欄位 | 最佳 2 卡值 | 說明 |
|--|--|--|
| Stage1 `attention_backend` / `mm_encoder_attn_backend` | `FLASHINFER` | 長 KV 必要 |
| Stage0/2/3 `max_num_seqs` | `4`（S1=`8`） | 配合 c=8 |
| Stage2 `attention_backend` | `FLASHINFER` | Talker FI |
| Stage2/3 `gpu_memory_utilization` | `0.15`（mem15） | 同卡擠 ASR |
| `codec_chunk_frames` | `20`（chunk20） | TTS chunk |
| Stage 放置 `devices` | `0`/`1`/`0`/`0` | 2 卡；4 卡用 split yaml |

改完 yaml **要重啟** server。

---

## 5. Bench 注意（OmniInteract）

1. 量測前必跑 **`0001` warmup**（同 server 進程）。  
2. 報表／精度／latency **排除 `0001`**。  
3. Remasure 已內建：`run_omni_c8_mm_flashinfer_rerun.sh`。

---

## 6. 快速決策

| 目標 | 做法 |
|--|--|
| 開箱即用（2 卡·功能齊） | `start_aura_omni.sh`（唔 set） |
| 2 卡 ASR 最優 | `VLLM_AURA_STAGE0_BYPASS=0` |
| 4 卡真 skip | `DEPLOY=…4gpu_split.yaml` + `CUDA_VISIBLE_DEVICES=0,1,2,3` |
| 更快首包音訊 | 保持 `SENTENCE_TTS=1`（預設） |
