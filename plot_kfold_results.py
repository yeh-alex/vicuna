import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 設定中文字型 (如果 Windows 系統支援)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  
plt.rcParams['axes.unicode_minus'] = False 

def plot_confusion_matrix(report_file):
    if not os.path.exists(report_file):
        print(f"[警告] 找不到檔案 {report_file}，請確認 K-Fold 訓練是否已百分之百跑完！")
        return
    
    df = pd.read_excel(report_file)
    y_true = df["Actual"]
    y_pred = df["Predict"]
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["不會再購 (0)", "會再購 (1)"], 
                yticklabels=["不會再購 (0)", "會再購 (1)"],
                annot_kws={"size": 14})
    plt.title("最佳模型獨立測試 - 混淆矩陣 (Confusion Matrix)", fontsize=14, pad=15)
    plt.xlabel("模型預測 (Predict)", fontsize=12)
    plt.ylabel("真實結果 (Actual)", fontsize=12)
    
    out_img = "confusion_matrix_kfold.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    print(f"✅ 已成功儲存混淆矩陣圖: {out_img}")
    plt.close()

def plot_loss_curves():
    # 尋找所有 fold 裡面的 trainer_state.json (存放在 checkpoint 中)
    folds = [d for d in os.listdir() if d.startswith("lora_fold_") and os.path.isdir(d)]
    if not folds:
        print("[警告] 找不到任何 lora_fold_ 目錄，無法繪製 Loss 曲線。")
        return
        
    plt.figure(figsize=(10, 6))
    
    # 配色與標籤
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, fold in enumerate(sorted(folds)):
        # 尋找這個 fold 裡面最新 / 數字最大的 checkpoint
        checkpoints = glob.glob(f"{fold}/checkpoint-*/trainer_state.json")
        if not checkpoints:
            continue
            
        # 抓取最後一個被建立的 json (擁有該 fold 最完整的訓練歷史)
        latest_json = sorted(checkpoints, key=os.path.getmtime)[-1]
        
        with open(latest_json, "r") as f:
            state = json.load(f)
            
        history = state.get("log_history", [])
        
        epochs_train = []
        loss_train = []
        epochs_eval = []
        loss_eval = []
        
        for log in history:
            if "loss" in log:
                epochs_train.append(log.get("epoch", 0))
                loss_train.append(log["loss"])
            if "eval_loss" in log:
                epochs_eval.append(log.get("epoch", 0))
                loss_eval.append(log["eval_loss"])
                
        # 繪製曲線
        fold_num = fold.split("_")[-1]
        color = colors[idx % len(colors)]
        
        if loss_train:
            plt.plot(epochs_train, loss_train, label=f"Fold {fold_num} - Train Loss", linestyle=":", color=color, alpha=0.7)
        if loss_eval:
            plt.plot(epochs_eval, loss_eval, label=f"Fold {fold_num} - Eval Loss", marker="o", color=color, linewidth=2)
            
    plt.title("K-Fold 訓練與驗證 Loss 曲線", fontsize=16, pad=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    
    # 將 Legend 移到外面以免擋住圖表
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    out_img = "training_curves_kfold.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    print(f"✅ 已成功儲存曲線圖: {out_img}")
    plt.close()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("📊 K-Fold 視覺化報表生成工具")
    print("="*50)
    plot_loss_curves()
    plot_confusion_matrix("final_report_kfold.xlsx")
    print("\n🎉 視覺化圖表繪製完成！請查看產生的 .png 檔案。")
