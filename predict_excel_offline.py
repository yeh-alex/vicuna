import os
# 必須在 import pytorch 或 pandas 之前設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

MODEL_ID = "lmsys/vicuna-7b-v1.5"
LORA_PATH = "./lora_checkpoints_0325/checkpoint-150"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def build_prompt(benevolence: float, integrity: float, competence: float, trust: float, tokenizer) -> str:
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
                f"Competence: {competence}, "
                f"Predict if this user will purchase again. "
                f"Output '1' if the user will buy again, or '0' if they will not. "
                f"Answer with ONLY the number (0 or 1)."
            )
        }
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser(description="離線批次預測 Excel 腳本 (支援 Power BI)")
    parser.add_argument("--input", "-i", type=str, default="LoRA_testing20260316.xlsx", help="輸入的 Excel 檔案路徑")
    parser.add_argument("--output", "-o", type=str, default="prediction_output.xlsx", help="輸出的 Excel 檔案路徑")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[錯誤] 找不到檔案: {args.input}")
        return

    print(f"--- 載入原始資料: {args.input} ---")
    try:
        df = pd.read_excel(args.input)
    except Exception as e:
        print(f"[錯誤] 讀取 Excel 失敗: {e}")
        return

    required_cols = ['B', 'I_reverse', 'C', 'trust']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[錯誤] Excel 缺少必要欄位: {missing}")
        return

    print("\n[LOAD] 初始化模型與 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.chat_template = VICUNA_TEMPLATE

    print("   [GPU] Loading with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    print("[OK] 模型載入完成！\n")

    print(f"--- 開始批次預測 (共 {len(df)} 筆資料) ---")
    predictions = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="預測進度"):
        prompt = build_prompt(row['B'], row['I_reverse'], row['C'], row['trust'], tokenizer)
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
            pred_int = int(prediction_text[0]) if prediction_text and prediction_text[0] in ['0', '1'] else 0
        except:
            pred_int = 0
            
        predictions.append(pred_int)
        labels.append("Purchase" if pred_int == 1 else "Not Purchase")

    # 將預測結果寫回原本的 DataFrame
    df['預測結果 (Prediction)'] = predictions
    df['預測標籤 (Label)'] = labels

    print("\n--- 準備生成 Power BI 優化報表 ---")
    # 產生 Summary 工作表提供給 Power BI 看板直接讀取（例如儀表板內的 KPI 卡片）
    summary_data = {
        "Metric": [
            "Total Customers (總客戶數)",
            "Predicted to Purchase (預測再購數)",
            "Predicted NOT to Purchase (預測流失數)",
            "Avg Benevolence (平均善意度)",
            "Avg Integrity (平均誠信度)",
            "Avg Competence (平均能力度)",
            "Avg Trust (平均信任度)",
            "Predicted Purchase Rate (預測再購率)"
        ],
        "Value": [
            len(df),
            sum(predictions),
            len(df) - sum(predictions),
            df['B'].mean(),
            df['I_reverse'].mean(),
            df['C'].mean(),
            df['trust'].mean(),
            f"{(sum(predictions) / len(df)) * 100:.2f}%"
        ]
    }
    df_summary = pd.DataFrame(summary_data)

    print(f"   儲存預測與報表至 {args.output} ...")
    with pd.ExcelWriter(args.output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Predictions(含原始資料)', index=False)
        df_summary.to_excel(writer, sheet_name='Summary Statistics(統計資料)', index=False)

        # 將欄寬微調讓 Excel / Power BI 看起來更整齊
        worksheet1 = writer.sheets['Predictions(含原始資料)']
        worksheet1.set_column('A:Z', 15)
        worksheet2 = writer.sheets['Summary Statistics(統計資料)']
        worksheet2.set_column('A:A', 35)
        worksheet2.set_column('B:B', 20)
        
    print("\n[OK] 處理完成！🎉")
    print(f"您可以直接將 {args.output} 載入您的 Power BI，它會自動為您分類好兩個實用工作表！")

if __name__ == "__main__":
    main()
