"""
Vicuna LoRA 3-Fold Cross-Validation 訓練腳本 (B, I_reverse, C, Trust)
==================================================
此腳本用來處理最新的 4000 筆 synthetic 資料，進行 3-Fold CV，解決潛在的 Data Leakage 問題。
並且在每個 Fold 完成後挑出表現最好的一個，進行最後獨立測資評估。

用法：
    conda activate vacuna_env
    python retrain_bict_kfold.py
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
from sklearn.metrics import accuracy_score, classification_report
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
TRAIN_FILE = "LoRA_training_4000_synthetic.xlsx"
TEST_FILE = "LoRA_testing20260316.xlsx"
MODEL_ID = "lmsys/vicuna-7b-v1.5"
N_SPLITS = 3
EPOCHS_PER_FOLD = 3
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
    """對獨立測試集或 Validation集做 Inference 並回傳 Accuracy"""
    model.eval()
    y_true, y_pred = [], []
    
    # 先做成 Dataset 只是為了一致性
    test_ds = create_dataset_from_df(test_df, tokenizer)
    
    for item in tqdm(test_ds, desc="Evaluating Accuracy"):
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
    print(f"啟動 {N_SPLITS}-Fold Cross-Validation 訓練與驗證")
    print("=" * 60)

    # 1. 讀取並檢查資料
    df_all = pd.read_excel(TRAIN_FILE)
    print(f"訓練集 `{TRAIN_FILE}` 總筆數: {len(df_all)}")
    print(f"再購(1) vs 不會再購(0):\n{df_all['再購'].value_counts().to_string()}")

    # Tokenizer 只要載入一次
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.chat_template = VICUNA_TEMPLATE

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    # 用於切分 K-Fold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    fold_accuracies = []
    best_fold = -1
    best_acc = 0.0

    # 執行迴圈
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_all, df_all['再購'])):
        print("\n" + "=" * 50)
        print(f"🔄 開始訓練 Fold {fold + 1} / {N_SPLITS}")
        print("=" * 50)

        # (1) 清理先前 Fold 的 GPU 記憶體 (非常重要，避免 OOM 和偷看)
        gc.collect()
        torch.cuda.empty_cache()

        # (2) 資料集切分與平衡
        df_train = df_all.iloc[train_idx].copy()
        df_val = df_all.iloc[val_idx].copy()
        
        # Train 集實行 Over-sampling 平衡
        df_train_balanced = upsample_df(df_train, '再購')
        print(f"[Fold {fold+1}] Train Size (Balanced): {len(df_train_balanced)}, Val Size: {len(df_val)}")

        # 轉成 HuggingFace Dataset
        train_ds = create_dataset_from_df(df_train_balanced, tokenizer)
        val_ds = create_dataset_from_df(df_val, tokenizer)

        tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["text", "label"])
        tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=["text", "label"])

        # (3) 每次重新初始化BaseModel 與 LoRA 權重
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
        fold_out_dir = f"./lora_fold_{fold+1}"
        training_args = TrainingArguments(
            output_dir=fold_out_dir,
            eval_strategy="epoch",  # 注意：在最新的 transformers 這裡已經改成 eval_strategy
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
        print(f"[Fold {fold+1}] 🚀 เริ่ม訓練 (Total Epochs: {EPOCHS_PER_FOLD})")
        trainer.train()

        # (6) 驗證 Accuracy
        print(f"[Fold {fold+1}] 進行 Validation Accuracy 測試...")
        val_acc, _, _ = evaluate_model(model, tokenizer, df_val)
        print(f">> Fold {fold+1} Validation Accuracy: {val_acc:.2%}")
        
        fold_accuracies.append(val_acc)
        
        # 儲存與記錄最佳 Fold
        if val_acc > best_acc:
            best_acc = val_acc
            best_fold = fold + 1
            model.save_pretrained("./vicuna_bict_lora_model_kfold")
            tokenizer.save_pretrained("./vicuna_bict_lora_model_kfold")
            print(f"⭐ Fold {fold+1} 打破紀錄，已將最好模型更新至 ./vicuna_bict_lora_model_kfold")

        # 訓練這折完畢，手動刪除相關變數確保下一折乾淨
        del model
        del base_model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # 結算與最終獨立點測試
    # ============================================================
    print("\n" + "#" * 60)
    print(f"🎉 {N_SPLITS}-Fold Cross-Validation 結算結果")
    print(f"各折 Accuracy: {fold_accuracies}")
    print(f"平均 Accuracy (真實驗證力): {np.mean(fold_accuracies):.2%}")
    print(f"挑選的最佳 Fold: Fold {best_fold} (Acc: {best_acc:.2%})")
    print("#" * 60)

    print(f"\n>>> 開始進行最終獨立測試: 載入最佳模型對決 {TEST_FILE}")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    final_model = PeftModel.from_pretrained(base_model, "./vicuna_bict_lora_model_kfold")
    
    df_test = pd.read_excel(TEST_FILE)
    test_acc, y_true, y_pred = evaluate_model(final_model, tokenizer, df_test)

    print("\n" + "=" * 50)
    print(f"🏆 Final Best Model Test Accuracy: {test_acc:.2%}")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["Not Purchase (0)", "Purchase (1)"]))

    # 匯出報告
    report_file = "final_report_kfold.xlsx"
    pd.DataFrame({
        "Actual": y_true, 
        "Predict": y_pred
    }).to_excel(report_file, index=False)
    
    print(f"\n>>> 獨立測試報告已儲存: {report_file}")
    print(">>> K-Fold 流程大功告成！新模型位於 ./vicuna_bict_lora_model_kfold")
