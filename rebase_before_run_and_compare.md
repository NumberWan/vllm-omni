# Rebase 前版本 (before-pr3232) 執行與對比指令

## 1) 確認目前分支/版本

```bash
git branch --show-current
git rev-parse --short HEAD
```

預期：
- branch: `before-pr3232`
- commit: `dfd9f4d9`（或同一時點）

## 2) 建議環境變數

```bash
export SNAP=/models/models--Qwen--Qwen-Image-2512/snapshots/25468b98e3276ca6700de15c6628e51b7de54a26
export SNAP1=$(ls -1d /models/models--QuantTrio--Qwen3-VL-30B-A3B-Instruct-AWQ/snapshots/* | tail -n1)

export HF_HOME=/models
export HUGGINGFACE_HUB_CACHE=/models
```

## 3) 產生 old parity log（手動 server + 單請求）

### 終端 A
```bash
export VLLM_OMNI_QWEN_PARITY_LOG=1
export VLLM_OMNI_QWEN_PARITY_LOG_FILE=/wtk/old/result/qwen_parity_old.log
python -m vllm_omni.entrypoints.cli.main serve "$SNAP" --host 127.0.0.1 --port 8093 --omni --stage-init-timeout 300 --num-gpus 1
```

### 終端 B（任一你想測的 prompt）
```bash
mkdir -p /wtk/old/result
curl -sS "http://127.0.0.1:8093/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"任务目标：创建数据库视图筛选器。界面描述：这是Windows电脑界面：页面上方有顶部工具栏，左侧可能有左侧栏；主体为表格/卡片列表。","n":1,"size":"1024x1024","seed":12345,"num_inference_steps":30}' \
  > /wtk/old/result/type4_chinese_computer_0002_old.json
```

## 4) 與 new parity log 對比

```bash
diff -u /wtk/old/result/qwen_parity_old.log /wtk/w00917303/result/qwen_parity_new.log > /wtk/old/result/parity_old_vs_new.diff
```

## 5) 測試產物目錄可以保留嗎？

可以。`tests/e2e/accuracy/artifacts/` 是未追蹤檔案，保留不影響 git 歷史。  
若要清掉：

```bash
rm -rf tests/e2e/accuracy/artifacts
```
