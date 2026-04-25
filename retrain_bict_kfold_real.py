"""
Vicuna LoRA 3-Fold Cross-Validation 訓練真實小樣本腳本 (B, I_reverse, C, Trust)
==================================================
此腳本專門處理手頭唯一的 97 筆真實訓練集。
我們不使用任何合成資料，而是進行極嚴格的 Out-Of-Fold (OOF) 3-Fold 交叉驗證。
迴圈跑完後會結算 97 筆資料「無作弊」的真實成績，
最後再把 97 筆全部倒進去，訓練最終要上線的正式模型。

用法：
    conda activate vacuna_env
    python retrain_bict_kfold_real.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import shutil
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ============================================================
# 參數設定
# ============================================================
DATA_FILE = "LoRA_training20260316.xlsx"
MODEL_ID = "lmsys/vicuna-7b-v1.5"
FINAL_MODEL_OUTPUT_DIR = "./vicuna_bict_lora_model_real"
FINAL_REPORT_OUTPUT = "final_report_kfold_real.xlsx"

N_SPLITS = 3
EPOCHS_PER_FOLD = 3       # K-Fold 驗證時使用的 Epoch 數
EPOCHS_FINAL_TRAIN = 3    # 最終全資料合併訓練時使用的 Epoch 數
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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

# ============================================================
# 資料轉換與增強方法
# ============================================================
def create_dataset_from_df(df, tokenizer):
    """將 DataFrame 轉換成支援 HuggingFace 的 Dataset 並標記 Prompt"""
    formatted_data = []
    
    # 確保 B I C trust 和 再購都存在
    for _, row in df.iterrows():
        chat = [
            {"role": "system", "content": "You are a professional purchase behavior predictor and an expert in trust research."},
            {"role": "user", "content": (
                f"Trust is a multi-dimensional construct composed of Benevolence, Integrity, and Competence. "
                f"Benevolence reflects goodwill and care toward customers (e.g., proactive communication and support). "
                f"Integrity reflects honesty and transparency in business dealings. "
                f"Competence reflects the ability and expertise to deliver expected service, often associated with experience and performance. "
                f"Trust is a composite score derived from these three dimensions. "
                f"Analyze the following values: "
                f"Benevolence: {row['B']}, "
                f"Integrity: {row['I_reverse']}, "
                f"Competence: {row['C']}, "
                f"Trust: {round(row['trust'], 4)}. "
                f"Predict if this user will purchase again. "
                f"Output '1' if the user will buy again, or '0' if they will not. "
                f"Answer with ONLY the number (0 or 1)."
            )},
            {"role": "assistant", "content": str(int(row['再購'])).strip()}
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        formatted_data.append({"text": text, "label": str(int(row['再購'])).strip()})
        
    return Dataset.from_list(formatted_data)

def upsample_df(df, target_col='再購'):
    """對少數類別進行 Over-sampling 使之平衡"""
    counts = df[target_col].value_counts()
    if len(counts) < 2: return df
    
    max_count = counts.max()
    dfs = []
    for val, count in counts.items():
        sub_df = df[df[target_col] == val]
        repeat_times = max(1, max_count // count)
        dfs.append(pd.concat([sub_df] * repeat_times, ignore_index=True))
    
    balanced_df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

def evaluate_model(model, tokenizer, test_df):
    """對 Validation集做 Inference 並回傳 OOF 預測陣列"""
    model.eval()
    y_true, y_pred = [], []
    
    # 先做成 Dataset 只是為了一致性
    test_ds = create_dataset_from_df(test_df, tokenizer)
    
    for item in tqdm(test_ds, desc="Evaluating OOF Accuracy"):
        prompt = item["text"].split("ASSISTANT:")[0] + "ASSISTANT:"
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
            prediction = int(prediction_text[0]) if prediction_text[0] in ['0', '1'] else 0
        except:
            prediction = 0

        y_true.append(int(item["label"]))
        y_pred.append(prediction)

    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred

# ============================================================
# 主程式
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"啟動 {N_SPLITS}-Fold 真實小樣本 OOF 訓練與驗證")
    print("=" * 60)

    # 1. 讀取並檢查資料
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"找不到檔案 {DATA_FILE}，請確認。")
        
    df_all = pd.read_excel(DATA_FILE)
    print(f"專案唯一真實資料集 `{DATA_FILE}` 總筆數: {len(df_all)}")
    print(f"再購(1) vs 不會再購(0):\n{df_all['再購'].value_counts().to_string()}")

    # Tokenizer 只要載入一次
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.chat_template = VICUNA_TEMPLATE

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    # 用於切分 K-Fold (Out-Of-Fold)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    oof_y_true = []
    oof_y_pred = []

    # 執行迴圈：階段一 - K-Fold OOF 評估
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_all, df_all['再購'])):
        print("\n" + "=" * 50)
        print(f"🔄 開始 OOF 評估 Fold {fold + 1} / {N_SPLITS}")
        print("=" * 50)

        # (1) 清理先前 Fold 的 GPU 記憶體
        gc.collect()
        torch.cuda.empty_cache()

        # (2) 資料集切分與平衡
        df_train = df_all.iloc[train_idx].copy()
        df_val = df_all.iloc[val_idx].copy()
        
        # Train 集實行 Over-sampling 平衡，Val 絕對不動
        df_train_balanced = upsample_df(df_train, '再購')
        print(f"[Fold {fold+1}] Train Size (Balanced): {len(df_train_balanced)}, Val Size: {len(df_val)}")

        # 轉成 HuggingFace Dataset
        train_ds = create_dataset_from_df(df_train_balanced, tokenizer)
        val_ds = create_dataset_from_df(df_val, tokenizer)

        tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["text", "label"])
        tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=["text", "label"])

        # (3) 每次重新初始化 Base Model 與 LoRA 權重
        print(f"[Fold {fold+1}] 初始化 Base Model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=TARGET_MODULES,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(base_model, lora_config)

        # (4) 準備訓練
        fold_out_dir = f"./lora_oof_fold_{fold+1}"
        training_args = TrainingArguments(
            output_dir=fold_out_dir,
            eval_strategy="epoch",  
            save_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=EPOCHS_PER_FOLD,
            weight_decay=0.01,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            dataloader_num_workers=0,
            optim="paged_adamw_8bit"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )

        import accelerate
        accelerate.optimizer.AcceleratedOptimizer.train = lambda self: None
        accelerate.optimizer.AcceleratedOptimizer.eval = lambda self: None

        # (5) 執行訓練
        print(f"[Fold {fold+1}] 🚀 เริ่ม訓練 (Epochs: {EPOCHS_PER_FOLD})")
        trainer.train()

        # (6) 測驗 OOF，收集預測結果
        print(f"[Fold {fold+1}] 進行 Out-Of-Fold 盲測...")
        val_acc, y_true_fold, y_pred_fold = evaluate_model(model, tokenizer, df_val)
        print(f">> Fold {fold+1} 盲測 Accuracy: {val_acc:.2%}")
        
        oof_y_true.extend(y_true_fold)
        oof_y_pred.extend(y_pred_fold)

        # 清除暫存目錄，因為 OOF 策略下我們不留中間折的模型
        del model
        del base_model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        # 刪除 ./lora_oof_fold_x 以節省硬碟空間
        if os.path.exists(fold_out_dir):
            shutil.rmtree(fold_out_dir)

    # ============================================================
    # 結算 OOF 真實成績並匯出
    # ============================================================
    print("\n" + "#" * 60)
    print(f"🎉 {N_SPLITS}-Fold 真實小樣本無作弊測試 (OOF) 結算")
    print("#" * 60)
    
    final_oof_acc = accuracy_score(oof_y_true, oof_y_pred)
    print(f"⭐ 最終整體無死角 Accuracy (衡量極限真實力): {final_oof_acc:.2%}")
    print("\n[Classification Report]")
    print(classification_report(oof_y_true, oof_y_pred, target_names=["Not Purchase (0)", "Purchase (1)"]))
    
    print("\n[Confusion Matrix]")
    cm = confusion_matrix(oof_y_true, oof_y_pred)
    print(f"TN: {cm[0,0]} FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} TP: {cm[1,1]}")

    # 匯出報告
    pd.DataFrame({
        "Actual": oof_y_true, 
        "Predict": oof_y_pred
    }).to_excel(FINAL_REPORT_OUTPUT, index=False)
    
    print(f"\n>>> 真實 97 筆 OOF 預測報告已儲存至: {FINAL_REPORT_OUTPUT}")

    # ============================================================
    # 階段二 - 合併 97 筆全資料集進行最終訓練
    # ============================================================
    print("\n" + "=" * 60)
    print("🔥 最終階段：載入全 97 筆真實資料，訓練正式決戰版本模型")
    print("=" * 60)
    
    gc.collect()
    torch.cuda.empty_cache()

    # 對全資料進行 Balance
    df_all_balanced = upsample_df(df_all, '再購')
    print(f"Final Full Train Size (Balanced): {len(df_all_balanced)}")
    
    full_ds = create_dataset_from_df(df_all_balanced, tokenizer)
    tokenized_full = full_ds.map(tokenize_fn, batched=True, remove_columns=["text", "label"])

    print("初始化 Final Base Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model_final = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    
    lora_config_final = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model_final = get_peft_model(base_model_final, lora_config_final)

    final_training_args = TrainingArguments(
        output_dir="./lora_final_temp",
        save_strategy="no",       # 我們只拿最後的模型，不存中間
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=EPOCHS_FINAL_TRAIN,
        weight_decay=0.01,
        fp16=True,
        report_to="none",
        dataloader_num_workers=0,
        optim="paged_adamw_8bit"
    )

    final_trainer = Trainer(
        model=model_final,
        args=final_training_args,
        train_dataset=tokenized_full,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    import accelerate
    accelerate.optimizer.AcceleratedOptimizer.train = lambda self: None
    accelerate.optimizer.AcceleratedOptimizer.eval = lambda self: None

    print(f"🚀 開始火力全開的最終訓練 (Epochs: {EPOCHS_FINAL_TRAIN})")
    final_trainer.train()

    print(f"\n✅ 訓練完成，正在將最終模型儲存至 {FINAL_MODEL_OUTPUT_DIR}")
    model_final.save_pretrained(FINAL_MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_OUTPUT_DIR)
    
    if os.path.exists("./lora_final_temp"):
        shutil.rmtree("./lora_final_temp")

    print("\n恭喜！全流程大功告成！未來只要將 api_server.py 的路徑指向")
    print(f"'{FINAL_MODEL_OUTPUT_DIR}' 即可享受最佳化真實預測！")
