import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm


ROLEPLAY_DATA_DIR = Path(os.environ.get("ROLEPLAY_DATA_DIR", Path(__file__).resolve().parents[1]))

a = os.environ.get("RAW_PARQUET", str(ROLEPLAY_DATA_DIR / "raw.parquet")) 
b = os.environ.get("SPE_PARQUET", str(ROLEPLAY_DATA_DIR / "spe.parquet"))  

# === 输入与输出路径 ===
files = [a, b]
output_file = os.environ.get("OUTPUT_PARQUET", str(ROLEPLAY_DATA_DIR / "test_1024.parquet"))  

# === 工具函数 ===
def get_task_from_filename(path: str):
    """从文件名中提取 task_name（例如 'math_eval.parquet' -> 'math_eval'）"""
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name

def update_extra_info(extra_info, new_idx):
    """更新样本内部 extra_info 的 index 字段"""
    if isinstance(extra_info, dict):
        extra_info["index"] = new_idx
    return extra_info


# === 合并逻辑 ===
all_data = []
skipped = 0

for file_path in files:
    task_name_from_file = get_task_from_filename(file_path)[0:3]
    print(f"🔍 读取 {file_path}，期望 task_name = {task_name_from_file}")
    
    df = pd.read_parquet(file_path)
    
    # 校验 task_name
    valid_rows = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Checking {task_name_from_file}"):
        try:
            row_task_name = None
            if "extra_info" in row and isinstance(row["extra_info"], dict):
                row_task_name = row["extra_info"].get("task_name", None)
            elif "task_name" in row:
                row_task_name = row["task_name"]

            if row_task_name != task_name_from_file:
                skipped += 1
                continue  # 跳过不匹配样本
            valid_rows.append(row)
        except Exception as e:
            skipped += 1
            continue
    
    if valid_rows:
        all_data.extend(valid_rows)

print(f"✅ 校验完成，过滤掉 {skipped} 条不匹配样本。")

# === 生成合并 DataFrame ===
merged_df = pd.DataFrame(all_data).reset_index(drop=True)

# === 重新编号 index ===
merged_df["index"] = range(len(merged_df))
if "extra_info" in merged_df.columns:
    merged_df["extra_info"] = [
        update_extra_info(ei, i) for i, ei in enumerate(merged_df["extra_info"])
    ]

# === 保存 ===
merged_df.to_parquet(output_file, index=False)
print(f"✅ 合并完成，共 {len(merged_df)} 条样本。输出文件：{output_file}")
