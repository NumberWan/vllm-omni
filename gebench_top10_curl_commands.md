# GEBench Top 10 Prompts - Curl Commands

使用方式：

1. 先確認你的 server 已經起在 `http://127.0.0.1:8093`。
2. 先建立輸出資料夾：`mkdir -p /wtk/w00917303/result`
3. 每條命令都會把 response 存成不同 JSON 檔（不會互相覆蓋）。

---

## 1) type4_chinese_computer_0001

```bash
curl -sS "http://127.0.0.1:8093/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"任务目标：重命名云文档。界面描述：这是Ubuntu电脑界面：页面上方有顶部工具栏，左侧可能有左侧栏；主体为表格/卡片列表。",
    "n":1,
    "size":"1024x1024",
    "seed":12345,
    "num_inference_steps":30
  }' > /wtk/w00917303/result/type4_chinese_computer_0001.json
```

## 2) type4_chinese_computer_0002

```bash
curl -sS "http://127.0.0.1:8093/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"任务目标：创建数据库视图筛选器。界面描述：这是Windows电脑界面：页面上方有顶部工具栏，左侧可能有左侧栏；主体为表格/卡片列表。",
    "n":1,
    "size":"1024x1024",
    "seed":12345,
    "num_inference_steps":30
  }' > /wtk/w00917303/result/type4_chinese_computer_0002.json
```

## 3) type4_chinese_computer_0003

```bash
curl -sS "http://127.0.0.1:8093/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"任务目标：将云文档移动到某文件夹。界面描述：这是macOS电脑界面：页面上方有顶部工具栏，左侧可能有左侧栏；主体为表格/卡片列表。",
    "n":1,
    "size":"1024x1024",
    "seed":12345,
    "num_inference_steps":30
  }' > /wtk/w00917303/result/type4_chinese_computer_0003.json
```

## 4) type4_chinese_computer_0004

```bash
curl -sS "http://127.0.0.1:8093/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"任务目标：创建过滤规则以自动归档邮件。界面描述：这是Windows电脑界面：页面上方有顶部工具栏，左侧可能有左侧栏；主体为表格/卡片列表。",
    "n":1,
    "size":"1024x1024",
    "seed":12345,
    "num_inference_steps":30
  }' > /wtk/w00917303/result/type4_chinese_computer_0004.json
```

## 5) type4_chinese_computer_0005

```bash
curl -sS "http://127.0.0.1:8093/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"任务目标：创建新标签并移动一封邮件。界面描述：这是Windows电脑界面：页面上方有顶部工具栏，左侧可能有左侧栏；主体为表格/卡片列表。",
    "n":1,
    "size":"1024x1024",
    "seed":12345,
    "num_inference_steps":30
  }' > /wtk/w00917303/result/type4_chinese_computer_0005.json
```

## 6) type4_chinese_computer_0006

```bash
curl -sS "http://127.0.0.1:8093/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"任务目标：分享云文档链接。界面描述：这是Windows电脑界面：页面上方有顶部工具栏，左侧可能有左侧栏；主体为表格/卡片列表。",
    "n":1,
    "size":"1024x1024",
    "seed":12345,
    "num_inference_steps":30
  }' > /wtk/w00917303/result/type4_chinese_computer_0006.json
```

## 7) type4_chinese_computer_0007

```bash
curl -sS "http://127.0.0.1:8093/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"任务目标：清除本次会话的收听历史。界面描述：这是Ubuntu电脑界面：页面上方有顶部工具栏，左侧可能有左侧栏；主体为表格/卡片列表。",
    "n":1,
    "size":"1024x1024",
    "seed":12345,
    "num_inference_steps":30
  }' > /wtk/w00917303/result/type4_chinese_computer_0007.json
```

## 8) type4_chinese_computer_0008

```bash
curl -sS "http://127.0.0.1:8093/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"任务目标：下载某区域以离线使用。界面描述：这是macOS电脑界面：页面上方有顶部工具栏，左侧可能有左侧栏；主体为表格/卡片列表。",
    "n":1,
    "size":"1024x1024",
    "seed":12345,
    "num_inference_steps":30
  }' > /wtk/w00917303/result/type4_chinese_computer_0008.json
```

## 9) type4_chinese_computer_0009

```bash
curl -sS "http://127.0.0.1:8093/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"任务目标：创建数据库视图筛选器。界面描述：这是Ubuntu电脑界面：页面上方有顶部工具栏，左侧可能有左侧栏；主体为表格/卡片列表。",
    "n":1,
    "size":"1024x1024",
    "seed":12345,
    "num_inference_steps":30
  }' > /wtk/w00917303/result/type4_chinese_computer_0009.json
```

## 10) type4_chinese_computer_0010

```bash
curl -sS "http://127.0.0.1:8093/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"任务目标：分享你的位置一小时。界面描述：这是macOS电脑界面：页面上方有顶部工具栏，左侧可能有左侧栏；主体为表格/卡片列表。",
    "n":1,
    "size":"1024x1024",
    "seed":12345,
    "num_inference_steps":30
  }' > /wtk/w00917303/result/type4_chinese_computer_0010.json
```

---

可选：把 JSON 转成 PNG（以第 0001 题为例）

```bash
python - <<'PY'
import json, base64
p="/wtk/w00917303/result/type4_chinese_computer_0001.json"
o="/wtk/w00917303/result/type4_chinese_computer_0001.png"
d=json.load(open(p,encoding="utf-8"))
open(o,"wb").write(base64.b64decode(d["data"][0]["b64_json"]))
print("saved:", o)
PY
```
