# Vicuna LoRA 再購預測 × n8n 串接指南

> 完整的操作指南，從零開始到成功串接。

---

## 📋 目錄

1. [系統架構](#系統架構)
2. [前置需求](#前置需求)
3. [Step 1：啟動 API Server](#step-1啟動-api-server)
4. [Step 2：安裝 n8n](#step-2安裝-n8n)
5. [Step 3：匯入 Workflow](#step-3匯入-workflow)
6. [Step 4：設定 Email 通知](#step-4設定-email-通知)
7. [Step 5：設定 Line 通知（可選）](#step-5設定-line-通知可選)
8. [Step 6：測試完整流程](#step-6測試完整流程)
9. [API 端點說明](#api-端點說明)
10. [常見問題 FAQ](#常見問題-faq)

---

## 系統架構

```
┌─────────────────┐     HTTP POST      ┌──────────────────┐
│                 │ ─────────────────► │                  │
│     n8n         │   /predict         │   FastAPI Server │
│  (localhost:    │ ◄───────────────── │  (localhost:8000)│
│    5678)        │   JSON Response    │   + Vicuna LoRA  │
│                 │                    │   (GPU)          │
└────────┬────────┘                    └──────────────────┘
         │
         ▼
  ┌──────────────┐
  │  Email 通知   │
  │  Line 通知    │
  └──────────────┘
```

---

## 前置需求

- ✅ Python 3.10+（你已經有 `vacuna_env`）
- ✅ NVIDIA GPU + CUDA（你已經有）
- ✅ 已訓練好的 LoRA 模型（`./vicuna_final_lora_model/`）
- ⬜ Node.js 18+（安裝 n8n 需要）

---

## Step 1：啟動 API Server

### 1.1 安裝 API 依賴

打開 **Anaconda Prompt** 或 **終端機**：

```bash
# 啟動你的虛擬環境
conda activate vacuna_env

# 進入專案目錄
cd C:\Users\alexy\OneDrive\桌面\vacuna

# 安裝 FastAPI 相關套件
pip install fastapi uvicorn python-multipart
```

### 1.2 啟動 Server

```bash
python api_server.py
```

你會看到：

```
==================================================
  Vicuna LoRA 再購預測 API Server
==================================================
  📍 API 文件：http://localhost:8000/docs
  📍 健康檢查：http://localhost:8000/health
==================================================
🚀 正在載入模型... (device: cuda)
   Base Model: lmsys/vicuna-7b-v1.5
   LoRA Path:  ./vicuna_bict_lora_model
✅ 模型載入完成！Server 準備就緒。
```

### 1.3 驗證 API 是否正常

打開**另一個終端**，執行：

```bash
curl http://localhost:8000/health
```

或直接用瀏覽器打開：**http://localhost:8000/docs**

你會看到互動式 API 文件，可以直接在網頁上測試！

---

## Step 2：安裝 n8n

### 2.1 安裝 Node.js

1. 前往 https://nodejs.org/
2. 下載 **LTS 版本**（例如 20.x）
3. 安裝時全部選預設即可

驗證安裝：
```bash
node --version   # 應顯示 v20.x.x
npm --version    # 應顯示 10.x.x
```

### 2.2 啟動 n8n

打開一個**新的 PowerShell / 終端機**（不要關掉 API Server 的那個）：

```bash
npx n8n
```

> ⚠️ 第一次執行會自動下載 n8n，需要幾分鐘。

啟動成功後會看到：

```
n8n ready on 0.0.0.0, port 5678
Editor is now accessible via: http://localhost:5678
```

打開瀏覽器前往 **http://localhost:5678**，第一次使用需要註冊一個本地帳號。

---

## Step 3：匯入 Workflow

1. 進入 n8n 介面
2. 點選左上角 **「+」** 或 **「Add workflow」**
3. 在新的 workflow 頁面中，點選右上角的 **「⋮」** 選單
4. 選擇 **「Import from file」**
5. 選擇 `C:\Users\alexy\OneDrive\桌面\vacuna\n8n_workflow.json`
6. Workflow 會自動載入所有節點

你會看到這樣的流程：

```
手動觸發 → 設定預測資料 → 呼叫預測 API → 判斷是否再購 → 發送 Email
```

---

## Step 4：設定 Email 通知

### 方法 A：使用 Gmail（推薦）

1. 先在 Google 帳號中開啟「應用程式密碼」：
   - 前往 https://myaccount.google.com/apppasswords
   - 選擇「郵件」→「Windows 電腦」
   - 複製產生的 16 位密碼

2. 在 n8n 中設定：
   - 點擊 **「發送 Email (會再購)」** 節點
   - 點擊 **Credential** 的「Create New」
   - 設定：
     - **User**: 你的 Gmail 地址
     - **Password**: 剛才的應用程式密碼
     - **Host**: `smtp.gmail.com`
     - **Port**: `465`
     - **SSL/TLS**: 開啟
   - 修改 `fromEmail` 和 `toEmail` 為你的 Email

3. 對 **「發送 Email (不會再購)」** 節點也做同樣設定（可以共用 Credential）

### 方法 B：使用其他 Email 服務

| 服務 | Host | Port |
|------|------|------|
| Gmail | smtp.gmail.com | 465 |
| Outlook | smtp.office365.com | 587 |
| Yahoo | smtp.mail.yahoo.com | 465 |

---

## Step 5：設定 Line 通知（可選）

### 5.1 使用 Line Notify

1. 前往 https://notify-bot.line.me/
2. 登入 → 點選「Generate Token」
3. 選擇要通知的聊天室
4. 複製 Token

### 5.2 在 n8n 中新增 Line 通知節點

在 workflow 中新增一個 **HTTP Request** 節點：

| 設定 | 值 |
|------|-----|
| Method | POST |
| URL | `https://notify-api.line.me/api/notify` |
| Header | `Authorization: Bearer YOUR_LINE_TOKEN` |
| Body (Form) | `message`: 預測結果文字 |

或者，你可以在 n8n 的 IF 節點之後，加一個 HTTP Request 節點來呼叫 Line Notify API：

```
Header:
  Authorization: Bearer YOUR_LINE_TOKEN_HERE

Body (Form Data):
  message: 再購預測結果：{{ $json.label }} (B={{ $json.benevolence }}, C={{ $json.competence }})
```

---

## Step 6：測試完整流程

### 6.1 確認清單

- [  ] API Server 正在運行（http://localhost:8000/health 回傳 ok）
- [  ] n8n 正在運行（http://localhost:5678 可以開啟）
- [  ] Workflow 已匯入
- [  ] Email Credential 已設定

### 6.2 執行測試

1. 在 n8n 中打開 workflow
2. 修改 **「設定預測資料」** 節點中的 benevolence 和 competence 值
3. 點選右上角 **「Test Workflow」** 或 **「Execute Workflow」**
4. 觀察每個節點是否亮綠燈
5. 檢查 Email 是否收到通知

---

## API 端點說明

### `GET /health` — 健康檢查

```bash
curl http://localhost:8000/health
```

回應：
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX ..."
}
```

### `POST /predict` — 單筆預測

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"benevolence": 6, "integrity": 8, "competence": 15, "trust": 4.2}'
```

回應：
```json
{
  "prediction": 0,
  "label": "Not Purchase",
  "raw_output": "0",
  "benevolence": 6.0,
  "integrity": 8.0,
  "competence": 15.0,
  "trust": 4.2
}
```

### `POST /predict/batch` — 批次預測

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"benevolence": 6, "integrity": 8, "competence": 15, "trust": 4.2},
      {"benevolence": 9, "integrity": 10, "competence": 18, "trust": 4.8}
    ]
  }'
```

### `POST /predict/excel` — 上傳 Excel 預測

```bash
curl -X POST http://localhost:8000/predict/excel \
  -F "file=@LoRA_testing20260316.xlsx"
```

---

## 常見問題 FAQ

### Q: 啟動 API Server 時出現 CUDA 錯誤？

確認你的 `vacuna_env` 可以使用 GPU：
```bash
conda activate vacuna_env
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: n8n 呼叫 API 時出現 Connection Refused？

1. 確認 API Server 是否在執行
2. 確認 URL 是 `http://localhost:8000/predict`（不是 https）
3. 嘗試在瀏覽器打開 http://localhost:8000/docs 確認

### Q: 模型載入很慢？

第一次載入需要下載 Vicuna-7B base model（約 4GB），之後會從快取載入，約 2-3 分鐘。

### Q: 如何讓 API Server 在後台執行？

```bash
# Windows PowerShell
Start-Process -NoNewWindow python -ArgumentList "api_server.py"
```

### Q: 如何停止 n8n 或 API Server？

在對應的終端按 `Ctrl + C` 即可。

---

## 快速啟動 Checklist

每次使用時，只需要：

```bash
# 終端 1：啟動 API
conda activate vacuna_env
cd C:\Users\alexy\OneDrive\桌面\vacuna
python api_server.py

# 終端 2：啟動 n8n
npx n8n
```

然後打開 http://localhost:5678 執行 workflow。

---

## 附加強大功能：離線 Excel 分析 (支援 Power BI)

如果您不需要即時的 n8n，只想**一次性分析大量的 Excel 表格**做月報，我們提供了一支離線腳本。

```bash
conda activate vacuna_env
conda run --no-capture-output -n vacuna_env python predict_excel_offline.py -i sample_bict_data.xlsx -o powerbi_output.xlsx
```

這支腳本會在背景直接啟動 AI 模型並分析整張表（只要確保有 `B`, `I_reverse`, `C`, `trust` 欄位），輸出包含 `Summary Statistics` 的全新報表，能讓您的 **Power BI** 直接匯入使用！
