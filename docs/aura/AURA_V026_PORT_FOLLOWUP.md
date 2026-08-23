# AURA_production_v026 — 缺口跟進

repo worktree: `/home/wtk/vllm-omni-AURA_026`  
branch: `AURA_production_v026`（base `origin/main` @ `5215e03a`）  
updated: 2026-08-10

本檔係移植後剩餘工作嘅跟進清單。以本檔為準；`/tmp/aura_port_v026/PORT_FOLLOWUP.md` 只係速睇副本。

---

## 1. 已完成

| 項 | 狀態 |
|----|------|
| L1 additive modules / deploy / demo | `b8283eb3` |
| L2 processors / TTS / ASR（保留 main `qwen3_tts` / `pipeline_registry`） | `62386882` |
| L3 video-stream（保留 main `serving_chat`） | `0acd3329` |
| L4 orchestrator / scheduler / worker / connectors | `d091c8ef`（pytest 213 passed） |
| L5 OmniInteract bench / docs / dual-demo / onboarding | `af4d1bc4`（pytest 125 passed） |
| venv | `.venv`：`vllm==0.26` + editable omni |
| GPU smoke `verify_e2e`（0001+0002，bypass=0） | PASS（text+audio） |
| Smoke3（被迫非 BEST knobs） | 見 §2 |
| **Stage0 bypass hang 代碼修復** | 見 §7（unit test 通過；live GPU 驗證待空卡） |

產物根目錄：`/tmp/aura_port_v026/`

---

## 2. Smoke3 現況（2026-08-07，非對等 BEST）

- Deploy：`/tmp/aura_port_v026/omniinteract_smoke3/aura_omni_2gpu_local.yaml`
- GPUs：`CUDA_VISIBLE_DEVICES=1,2`
- **Knobs（非 BEST）**：`VLLM_AURA_STAGE0_BYPASS=0`、`VLLM_AURA_SENTENCE_TTS=0`
- Warmup：`0001`（排除計分）；計分：`0002–0004`

| Spoken（n=24） | Median |
|----------------|-------:|
| TTFT | 195 ms |
| TPOT | 5.8 ms |
| E2EL | 832 ms |
| AUDIO_TTFP | 132 ms |
| Stage1 TTFT | 78 ms |

| QA（soft-match） | 值 |
|------------------|-----:|
| Evaluated slots | 12 |
| IA-QTF1 | 0.1394 |

報表：`/tmp/aura_port_v026/omniinteract_smoke3/results_full/bench_report.md`

對照舊 Omni c1 BEST-ish：spoken median TTFT ~139 ms、Stage1 ~72 ms。

---

## 3. 已知結論：IA-QTF1≈0.14 唔係移植回歸

- 舊 Omni Smoke3 baseline：`IA-QTF1≈0.1391`
- Native c1 離線 soft-match：`≈0.11`
- 12 slot 全部 `tp_core=0`（slot 窗口 + substring soft-match）
- **唔用 soft-match IA-QTF1 當精度通過門檻**

---

## 4. 待做 checklist

- [x] 修 Stage0 bypass hang（代碼 + unit test）— 見 §7
- [ ] **Live 驗證** `BYPASS=1` 短測唔 hang（需空閒 GPU；2026-08-10 全卡被佔）
- [ ] BEST knobs Smoke3 重測（`BYPASS=1` + `TTS_GATE=1` + `SENTENCE_TTS=1`；0001 warmup 排除計分）
- [ ] 更新本檔 §8 Smoke3 BEST 數字
- [ ] （可選）漏答對照舊 baseline — 只記錄

---

## 5. 唔硬做

| 項 | 原因 |
|----|------|
| 改 IA-QTF1 slot／soft-match | 評分設計，唔係 port bug |
| 接 LLM judge | 無現成管線 |
| 為 soft-match 分數改 serving | 目標錯 |
| 覆蓋 main `serving_chat` / `config_factory` / 整檔舊 `qwen3_tts` | 刻意保留 0.26 |
| 無 2 空卡時硬報「性能已追平」 | 需 BEST knobs 實測 |

---

## 6. 量測 knobs 對照

| Knob | 預設 BEST | 2026-08-07 Smoke3 | 目標重測 |
|------|-----------|-------------------|----------|
| `VLLM_AURA_STAGE0_BYPASS` | 1 | **0** | **1** |
| `VLLM_AURA_TTS_GATE_ON_VOICE_ASR` | 1 | 1 | 1 |
| `VLLM_AURA_SENTENCE_TTS` | 1 | **0** | **1** |
| Warmup / 計分 | 0001 / 0002–0004 | 同左 | 同左 |

---

## 7. Bypass hang — 根因與修復

| 欄位 | 內容 |
|------|------|
| Status | **fixed in code**（live GPU verify **blocked** — 無空卡） |
| Symptom | `BYPASS=1` + `async_chunk` 喺 `Bypassing stage-0` / `Created connector` 後掛死 |
| Evidence | `/tmp/aura_port_v026/omniinteract_smoke3/server_20260807_141306.log` @ 14:23:15 → 21min 無 Stage1 |
| Root cause | v0.26 `get_stage_connector_spec` 對只出現喺 `from_stage` 嘅 Stage1 設 `role=sender` → `_stage_receives_async_chunks(1)=False` → prewarm **跳過 Stage1**，但 bypass 仍 **SHM inject** → orphan chunk、Stage1 從未 admission |
| Fix | [`orchestrator.py`](../../vllm_omni/engine/orchestrator.py) `_bypass_stage0`：若 Stage1 唔係 async-chunk receiver，改 **mock-forward**（orchestrator 餵 Stage1）；仍 prewarm Stage2/3 |
| Test | `tests/engine/test_orchestrator_stage_bypass.py::test_stage0_bypass_mock_forwards_when_stage1_is_sender_only` + 既有 2 條 → **3 passed** |
| Live verify | 2026-08-10：GPU0 被 `yueqian` StageEngine 佔用；GPU1–3 亦滿。無法起 server。空卡後：`ALLOW_ONE_GPU=1` + 1gpu demo yaml 或 2GPU best，`BYPASS=1`，跑 silent vision turn / Smoke3 |

---

## 8. Smoke3 BEST（重測後填）

| Spoken median | 值 |
|---------------|-----|
| TTFT | —（blocked：無空閒 GPU） |
| TPOT | — |
| AUDIO_TTFP | — |
| Stage1 TTFT | — |
| IA-QTF1（只記錄） | — |
| Artifacts | — |

重測指令（空卡後）：

```bash
cd /home/wtk/vllm-omni-AURA_026
export CUDA_VISIBLE_DEVICES=0,1   # 或實際空卡
export DEPLOY=/tmp/aura_port_v026/omniinteract_smoke3/aura_omni_2gpu_local.yaml
export VLLM_AURA_STAGE0_BYPASS=1
export VLLM_AURA_TTS_GATE_ON_VOICE_ASR=1
export VLLM_AURA_SENTENCE_TTS=1
export LOG_DIR=/tmp/aura_port_v026/omniinteract_smoke3_best
bash scripts/start_aura_omni.sh
# 然後 bench：omniinteract_video_ids=0001,0002,0003,0004；報表排除 0001
```
