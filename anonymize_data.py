import os
import glob
import pandas as pd

def main():
    print("================== 處理去識別化與整合資料 ==================")
    print("1. 掃描 cus_data 中的所有銷售紀錄...")
    
    all_files = glob.glob(os.path.join("cus_data", "*.xlsx"))
    if not all_files:
        print("[錯誤] 在 cus_data 裡面找不到任何 Excel 檔案！")
        return
        
    df_list = []
    for file in all_files:
        print(f"   讀取: {file}")
        try:
            # 讀取發票檔案 (指定 openpyxl 防止找不到 engine)
            df = pd.read_excel(file, engine='openpyxl')
            df_list.append(df)
        except Exception as e:
            print(f"   [錯誤] 讀取 {file} 失敗: {e}")
            
    # 將所有年份的發票資料合併成一個巨大名單
    master_df = pd.concat(df_list, ignore_index=True)
    print(f">> 成功合併歷史發票！總計 {len(master_df)} 筆紀錄。")
    
    # 清理欄位與建立對應字典
    # 我們假設原始資料裡有包含「客戶名稱」和「客戶代號」
    # 若有意外的空白字元 (像是 "王大明   ")，我們會用 .str.strip() 去除空白以便建立準確字典
    print("\n2. 建立『客戶名稱』<->『客戶代號』查找字典...")
    
    if "客戶名稱" in master_df.columns and "客戶代號" in master_df.columns:
        master_df["客戶名稱"] = master_df["客戶名稱"].astype(str).str.strip()
        
        # 將客戶名稱與代號獨立出來
        mapping_df = master_df[["客戶名稱", "客戶代號"]].dropna().drop_duplicates()
        name_to_id = dict(zip(mapping_df["客戶名稱"], mapping_df["客戶代號"]))
        print(f">> 成功建立對照字典，共有 {len(name_to_id)} 位獨立客戶。")
    else:
        print("[錯誤] master_df 缺少『客戶名稱』或『客戶代號』欄位。列出所有欄位:", master_df.columns.tolist())
        return

    # 處理預測測試資料
    print("\n3. 將測試集進行去識別化 (移除 Name，改為 客戶代號 ID)...")
    test_file = "LoRA_testing20260316.xlsx"
    if os.path.exists(test_file):
        test_df = pd.read_excel(test_file, engine='openpyxl')
        if "Name" in test_df.columns:
            test_df["Name"] = test_df["Name"].astype(str).str.strip()
            # 利用我們剛剛建立的對照字典去反查每個人對應的客戶代號
            # 若無對應代號則標為 -1 (或 Nan)
            test_df["客戶代號"] = test_df["Name"].map(name_to_id)
            
            # 移除舊有 Name 欄位，確保資料高度匿名
            test_df = test_df.drop(columns=["Name"])
            
            # 將『客戶代號』移到 Dataframe 的最前面第一欄比較好看
            cols = ["客戶代號"] + [c for c in test_df.columns if c != "客戶代號"]
            test_df = test_df[cols]
            
            out_test = "LoRA_testing_demo_ready.xlsx"
            test_df.to_excel(out_test, index=False, engine='openpyxl')
            print(f">> 成功！已輸出完全匿名的測試集 => {out_test}")
        else:
            print(f"[警告] {test_file} 缺少 Name 欄位，無法進行轉換。")
    else:
        print(f"[警告] 找不到 {test_file}。")
        
    # 處理最終的 Master Sales DB
    print("\n4. 移除歷史資料總表中所有的身分識別資訊 (客戶名稱)...")
    if "客戶名稱" in master_df.columns:
        master_df = master_df.drop(columns=["客戶名稱"])
        
    out_master = "master_sales_demo_ready.xlsx"
    master_df.to_excel(out_master, index=False, engine='openpyxl')
    print(f">> 成功！已輸出完全匿名的 PowerBI 總表來源 => {out_master}")
    
    print("\n[OK] 去識別化工程全數完畢！🎉")

if __name__ == "__main__":
    main()
