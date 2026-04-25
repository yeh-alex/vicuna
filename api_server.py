"""
Vicuna LoRA 再購預測 API Server
================================
將微調好的 Vicuna-7B + LoRA 模型包裝成 REST API，供 n8n 呼叫。

啟動方式：
    python api_server.py

API 文件：
    啟動後開啟 http://localhost:8000/docs 可看到互動式文件
"""

import os

# 修復 OpenMP 重複載入問題（conda 環境常見）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import torch
import pandas as pd
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# 設定
# ============================================================
MODEL_ID = "lmsys/vicuna-7b-v1.5"
LORA_PATH = "./lora_checkpoints_0325/checkpoint-150"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全域變數（在 lifespan 中初始化）
model = None
tokenizer = None


# ============================================================
# Vicuna Chat Template（與訓練時一致）
# ============================================================
VICUNA_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] + ' ' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ 'USER: ' + message['content'] + ' ' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'ASSISTANT: ' + message['content'] + '</s>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'ASSISTANT:' }}"
    "{% endif %}"
)


def build_prompt(benevolence: float, integrity: float, competence: float, trust: float) -> str:
    """
    建立與訓練時完全一致的 prompt 格式。
    """
    chat = [
        {
            "role": "system",
            "content": "You are a professional purchase behavior predictor and an expert in trust research."
        },
        {
            "role": "user",
            "content": (
                f"Trust is a multi-dimensional construct composed of Benevolence, Integrity, and Competence. "
                f"Benevolence reflects goodwill and care toward customers (e.g., proactive communication and support). "
                f"Integrity reflects honesty and transparency in business dealings. "
                f"Competence reflects the ability and expertise to deliver expected service, often associated with experience and performance. "
                f"Trust is a composite score derived from these three dimensions. "
                f"Analyze the following values: "
                f"Benevolence: {benevolence}, "
                f"Integrity: {integrity}, "
                f"Competence: {competence}, "
                f"Trust: {round(trust, 4)}. "
                f"Predict if this user will purchase again. "
                f"Output '1' if the user will buy again, or '0' if they will not. "
                f"Answer with ONLY the number (0 or 1)."
            )
        }
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def run_inference(benevolence: float, integrity: float, competence: float, trust: float) -> dict:
    """
    執行單筆推論。
    """
    prompt = build_prompt(benevolence, integrity, competence, trust)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    prediction_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    try:
        prediction = int(prediction_text[0]) if prediction_text and prediction_text[0] in ['0', '1'] else 0
    except (IndexError, ValueError):
        prediction = 0

    return {
        "prediction": prediction,
        "label": "Purchase" if prediction == 1 else "Not Purchase",
        "raw_output": prediction_text,
        "benevolence": benevolence,
        "integrity": integrity,
        "competence": competence,
        "trust": trust
    }


# ============================================================
# FastAPI 應用
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用啟動時載入模型，關閉時釋放資源。"""
    global model, tokenizer

    # 詳細的系統診斷
    print(f"[INFO] System check:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version:    {torch.version.cuda}")
        print(f"   GPU:             {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory:      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"   [WARN] No CUDA GPU detected! Using CPU mode (slower)")
        print(f"   [TIP]  For GPU acceleration, install CUDA PyTorch:")
        print(f"      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    print(f"\n[LOAD] Loading model... (device: {DEVICE})")
    print(f"   Base Model: {MODEL_ID}")
    print(f"   LoRA Path:  {LORA_PATH}")

    # 載入 tokenizer（從 base model 載入，避免版本相容性問題）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.chat_template = VICUNA_TEMPLATE

    # 根據 CUDA 是否可用，選擇不同的載入方式
    if DEVICE == "cuda":
        # GPU 模式：使用 4-bit 量化（快、省記憶體）
        print("   [GPU] Loading with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        # CPU 模式：不量化，使用 float32（慢但能跑）
        print("   [CPU] Loading without quantization (~14GB RAM needed)...")
        print("   [WAIT] This will be slow, please wait...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

    # 載入 LoRA adapter
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    mode_label = "GPU (4-bit)" if DEVICE == "cuda" else "CPU (fp32, slower)"
    print(f"\n[OK] Model loaded! Mode: {mode_label}")
    print(f"   Server is ready.")

    yield  # 應用運行中

    # 關閉時清理
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[STOP] Model unloaded, server closed.")


app = FastAPI(
    title="Vicuna LoRA 再購預測 API",
    description=(
        "基於 Vicuna-7B + LoRA 微調的客戶再購預測模型 API。\n\n"
        "輸入 Benevolence（善意度）、Integrity（誠信度）、Competence（能力度）和 Trust（信任度）數值，\n"
        "預測客戶是否會再次購買。"
    ),
    version="1.0.0",
    lifespan=lifespan
)

# 允許跨域（讓 n8n 能呼叫）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request / Response 模型
# ============================================================
class PredictRequest(BaseModel):
    benevolence: float = Field(..., description="善意度 (Benevolence) 數值", examples=[6])
    integrity: float = Field(..., description="誠信度 (Integrity) 數值", examples=[8])
    competence: float = Field(..., description="能力度 (Competence) 數值", examples=[15])
    trust: float = Field(..., description="信任度 (Trust) 數值", examples=[4.2])

class PredictResponse(BaseModel):
    prediction: int = Field(..., description="預測結果：1=再購, 0=不再購")
    label: str = Field(..., description="預測標籤")
    raw_output: str = Field(..., description="模型原始輸出")
    benevolence: float = Field(..., description="輸入的善意度")
    integrity: float = Field(..., description="輸入的誠信度")
    competence: float = Field(..., description="輸入的能力度")
    trust: float = Field(..., description="輸入的信任度")

class BatchPredictRequest(BaseModel):
    items: List[PredictRequest] = Field(..., description="批次預測資料列表")

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int = Field(..., description="總筆數")
    purchase_count: int = Field(..., description="預測會再購的筆數")
    not_purchase_count: int = Field(..., description="預測不會再購的筆數")

class PowerBIRequest(BaseModel):
    customer_id: str = Field(..., description="要呼叫的客戶代號 (ID)")
    report_url: str = Field(
        "https://app.powerbi.com/groups/me/reports/YOUR_REPORT_ID/ReportSection", 
        description="PowerBI 網頁版連結"
    )
    table_name: str = Field(
        "master_sales_demo_ready", 
        description="PowerBI 內的 Dataset 資料表名稱"
    )


# ============================================================
# API Endpoints
# ============================================================
@app.get("/health", tags=["系統"])
async def health_check():
    """健康檢查 — 確認 Server 是否正常運行、模型是否已載入。"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.post("/predict", response_model=PredictResponse, tags=["預測"])
async def predict(request: PredictRequest):
    """
    單筆再購預測。

    輸入 Benevolence 和 Competence 數值，回傳預測結果。
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未載入完成，請稍後再試。")

    result = run_inference(request.benevolence, request.integrity, request.competence, request.trust)
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["預測"])
async def predict_batch(request: BatchPredictRequest):
    """
    批次再購預測。

    一次送入多筆資料，回傳所有預測結果及統計摘要。
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未載入完成，請稍後再試。")

    results = []
    for item in request.items:
        result = run_inference(item.benevolence, item.integrity, item.competence, item.trust)
        results.append(result)

    purchase_count = sum(1 for r in results if r["prediction"] == 1)

    return {
        "results": results,
        "total": len(results),
        "purchase_count": purchase_count,
        "not_purchase_count": len(results) - purchase_count
    }


@app.post("/predict/excel", tags=["預測"])
async def predict_from_excel(file: UploadFile = File(...)):
    """
    上傳 Excel 檔案進行批次預測。

    Excel 檔案必須包含 'B'（Benevolence）和 'C'（Competence）欄位。
    回傳每筆資料的預測結果。
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未載入完成，請稍後再試。")

    # 讀取 Excel
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法讀取 Excel 檔案：{str(e)}")

    # 檢查必要欄位
    if 'B' not in df.columns or 'C' not in df.columns or 'I_reverse' not in df.columns or 'trust' not in df.columns:
        available_cols = list(df.columns)
        raise HTTPException(
            status_code=400,
            detail=f"Excel 必須包含 'B', 'I_reverse', 'C', 'trust' 欄位。目前欄位：{available_cols}"
        )

    # 逐筆預測
    results = []
    for _, row in df.iterrows():
        result = run_inference(float(row['B']), float(row['I_reverse']), float(row['C']), float(row['trust']))
        if 'Name' in df.columns:
            result["name"] = str(row['Name'])
        results.append(result)

    purchase_count = sum(1 for r in results if r["prediction"] == 1)

    return {
        "results": results,
        "total": len(results),
        "purchase_count": purchase_count,
        "not_purchase_count": len(results) - purchase_count,
        "filename": file.filename
    }


@app.post("/open_powerbi", tags=["整合"])
async def open_powerbi_report(request: PowerBIRequest):
    """
    接收 n8n 傳來的客戶 ID，並自動拼接 PowerBIDashboard 的 URL 參數 (filter)，
    接著在本機預設瀏覽器中彈出該已過濾特定客人的資料表畫面。
    """
    import webbrowser
    
    # 組合 PowerBI filter 參數 (注意如果 ID 裡面沒有特殊符號，可以直接用字串)
    # 格式 ?filter=TableName/ColumnName eq 'Value'
    target_url = f"{request.report_url}?filter={request.table_name}/客戶代號 eq '{request.customer_id}'"
    
    # 呼叫系統預設瀏覽器打開這段帶有參數過濾條件的連結
    webbrowser.open(target_url)
    
    return {"status": "success", "opened_url": target_url}


# ============================================================
# 啟動 Server
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  Vicuna LoRA 再購預測 API Server")
    print("=" * 50)
    print(f"  -> API docs:     http://localhost:8000/docs")
    print(f"  -> Health check: http://localhost:8000/health")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
