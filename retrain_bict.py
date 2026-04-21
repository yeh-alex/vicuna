"""
Vicuna LoRA 重新訓練腳本（B, I, C, Trust 四維度）
==================================================
將 Prompt 從只有 B, C 擴展為 B, I_reverse, C, Trust 四個維度。
新模型儲存至 ./vicuna_bict_lora_model/

用法：
    conda activate vacuna_env
    python retrain_bict.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ============================================================
# 設定
# ============================================================
TRAIN_FILE = "synthetic_train_BICT_repurchase.xlsx"
TEST_FILE = "synthetic_test_BICT_repurchase.xlsx"
MODEL_ID = "lmsys/vicuna-7b-v1.5"
OUTPUT_DIR = "./vicuna_bict_lora_model_v2"


# ============================================================
# 資料處理（新增 Integrity 和 Trust）
# ============================================================
def process_data_to_chat(file_path, tokenizer):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find file: {file_path}")

    df = pd.read_excel(file_path)
    formatted_data = []

    for _, row in df.iterrows():
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
                    f"Benevolence: {row['B']}, "
                    f"Integrity: {row['I_reverse']}, "
                    f"Competence: {row['C']}, "
                    f"Trust: {round(row['trust'], 4)}. "
                    f"Predict if this user will purchase again. "
                    f"Output '1' if the user will buy again, or '0' if they will not. "
                    f"Answer with ONLY the number (0 or 1)."
                )
            },
            {"role": "assistant", "content": str(int(row['再購'])).strip()}
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        formatted_data.append({"text": text, "label": str(int(row['再購'])).strip()})

    return Dataset.from_list(formatted_data)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Device: {device}")
    if device == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ============================================================
    # Tokenizer
    # ============================================================
    print("\n>>> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    vicuna_template = (
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
    tokenizer.chat_template = vicuna_template

    # ============================================================
    # 準備資料集
    # ============================================================
    print("\n>>> Preparing datasets (B, I, C, Trust)...")
    df_train = pd.read_excel(TRAIN_FILE)
    print(f"    Training data distribution:")
    print(f"    {df_train['再購'].value_counts().to_dict()}")

    full_train_dataset = process_data_to_chat(TRAIN_FILE, tokenizer)
    dataset_split = full_train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    # 過採樣少數類，讓訓練集平衡
    label_0 = train_dataset.filter(lambda x: x["label"] == "0")
    label_1 = train_dataset.filter(lambda x: x["label"] == "1")
    if len(label_0) >= len(label_1) and len(label_1) > 0:
        repeat_times = max(1, len(label_0) // len(label_1))
        label_1_upsampled = concatenate_datasets([label_1] * repeat_times)
        train_dataset = concatenate_datasets([label_0, label_1_upsampled]).shuffle(seed=42)
    elif len(label_1) > len(label_0) and len(label_0) > 0:
        repeat_times = max(1, len(label_1) // len(label_0))
        label_0_upsampled = concatenate_datasets([label_0] * repeat_times)
        train_dataset = concatenate_datasets([label_0_upsampled, label_1]).shuffle(seed=42)

    print(f"    Balanced: label 0: {train_dataset['label'].count('0')}, label 1: {train_dataset['label'].count('1')}")

    test_dataset_raw = process_data_to_chat(TEST_FILE, tokenizer)

    # 顯示一個範例 prompt
    print(f"\n>>> Sample prompt:")
    print(full_train_dataset[0]['text'][:500])
    print("...")

    # ============================================================
    # Tokenize
    # ============================================================
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text", "label"])
    tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text", "label"])

    # ============================================================
    # 載入模型 + LoRA
    # ============================================================
    print("\n>>> Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ============================================================
    # 訓練
    # ============================================================
    print("\n>>> Starting Fine-tuning (B, I, C, Trust)...")
    training_args = TrainingArguments(
        output_dir="./lora_checkpoints_bict_v2",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=10,
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

    # Monkey-patch accelerate optimizer to fix AttributeError
    import accelerate
    accelerate.optimizer.AcceleratedOptimizer.train = lambda self: None
    accelerate.optimizer.AcceleratedOptimizer.eval = lambda self: None

    trainer.train()

    # 儲存模型
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n>>> Model saved to {OUTPUT_DIR}")

    # ============================================================
    # 測試集評估
    # ============================================================
    print("\n>>> Running test set evaluation...")
    model.eval()
    y_true, y_pred = [], []

    for i, item in enumerate(tqdm(test_dataset_raw, desc="Evaluating")):
        prompt = item["text"].split("ASSISTANT:")[0] + "ASSISTANT:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        prediction_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if i < 10:
            print(f"  [{i}] raw output: {repr(prediction_text)} | label: {item['label']}")

        try:
            prediction = prediction_text[0] if prediction_text[0] in ['0', '1'] else "0"
        except:
            prediction = "0"

        y_true.append(item["label"])
        y_pred.append(prediction)

    # 輸出結果
    print("\n" + "=" * 50)
    print(f"  BICT Model - Final Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print(f"  (Previous BICT model for comparison)")
    print("  Label: 1 = Will Purchase, 0 = Will Not Purchase")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["Not Purchase (0)", "Purchase (1)"]))

    # 匯出結果
    report_file = "final_report_bict_v2.xlsx"
    pd.DataFrame({"Actual": y_true, "Predict": y_pred}).to_excel(report_file, index=False)
    print(f">>> Report saved: {report_file}")
    print("\n>>> Done! You can now update api_server.py to use the new model.")

