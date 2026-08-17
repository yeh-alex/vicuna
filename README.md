# Vicuna BICT 再購預測系統 (Vicuna BICT Repurchase Prediction)

## 專案簡介
這是一套基於大型語言模型 (LLM) 微調技術的智能決策系統。我們使用 `lmsys/vicuna-7b-v1.5` 為基礎模型，加上獨家領域資料進行深度訓練（LoRA Fine-Tuning），使它能根據顧客的**高階信任度四大維度 (BICT)**，預測該顧客未來是否會再次購買。

### 四大推論維度指標 (BICT)：
1. **Benevolence (善意度/B)**：品牌是否展現對顧客的關心與友善支援。
2. **Integrity (誠信度/I)**：商業行為是否誠實、透明、言出必行。
3. **Competence (能力度/C)**：服務或產品的真實效能與專業程度。
4. **Trust (信任度/Trust)**：由上述三項綜合出的核心感受。

模型會根據這四個分數自動進行交叉推論，並給出二元決策結論：**會再購 (Purchase)** 或 **不會再購 (Not Purchase)**。

---

## 核心功能與應用場景

本專案支援兩種截然不同的運行模式，從單筆即時自動化到大數據報表產出皆能涵蓋：

### 1. 即時 API 服務 (`api_server.py`)
打造專屬的 AI 推論後台，提供標準化 RESTful API 供前端或第三方服務接入。透過啟動 `api_server.py`，你可以在本機架構一台 GPU 驅動的預測伺服器。
- 支援單筆即時推論 **`/predict`** 與批次推論 **`/predict/batch`**
- 開箱即用的互動式 API 文件 (Swagger) 於 `http://localhost:8000/docs`

### 2. n8n 流程自動化 (`n8n_workflow.json`)
可結合開源自動化工具 **n8n** 。當有新客戶的回饋問卷資料流入時，自動透過 API 請求 AI 預測。若模型判斷該客戶「不會再購」，系統可自動觸發並寄送含有優惠券的 Email 或發送 Line 通知給客服。
- **詳細教學請見：[n8n 完整串接指南](README_n8n.md)**

### 3. 離線大數據批次分析與 Power BI 整合 (`predict_excel_offline.py`)
專為行銷數據與商業分析設計。只要輸入你的 `Excel` 表單，即可自動批次處理數千筆客戶分數。系統不只幫你填上預測結果，還會額外產出「總結統計 (Summary Statistics)」工作表。產生完的報表可以直接丟給 **Power BI** 製作專業級儀表板。
- **詳細圖文說明：[給各部門的快速教學與案例展示](tutorial_for_others.md)**

### 4. 自動化模型持續訓練 (`retrain_bict.py` & `LoRA_v2.ipynb`)
包含了模型的完整訓練管道。採用 HuggingFace `peft` 與 `transformers` 框架，隨時可匯入最新蒐集的客戶資料 (`.xlsx`) 繼續迭代您的 LoRA 模型。腳本也會自動產出包含混淆矩陣、Accuracy 曲線等數據分析報表，確保預測品質。

---

## 快速開始

### 系統需求
- Python 3.10+
- NVIDIA GPU (建議至少 8GB+ VRAM 以執行 4-bit 量化)
- Conda 虛擬環境配置

### 啟動服務測試
**啟動 API 伺服器：**
```bash
conda activate vacuna_env
python api_server.py
```
> Server 啟動後，開啟瀏覽器前往 `http://localhost:8000/docs` 即可直接網頁介面上進行預測測試！

**執行 Excel 批次預測 (免 API)：**
```bash
# 此模式不需啟動 api_server
conda activate vacuna_env
python predict_excel_offline.py -i sample_bict_data.xlsx -o output_result.xlsx
```

---

## 常見問題
如果於執行過程中遇到 GPU、環境依賴或 n8n 串接問題，請善用 [README_n8n.md](README_n8n.md) 的 FAQ 區塊查詢解決方案。
