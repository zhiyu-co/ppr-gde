import pandas as pd

# === 抽样数量 ===
n_samples = 2048

# === 输入与输出路径 ===
input_file = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/char_rm_4096.parquet"       # 原始 parquet 文件路径
output_file = f"/gemini/space/private/cgn/project/cllm_rl/data/roleplay/char_rm_{n_samples}.parquet"   # 输出样本文件路径



# === 读取原文件 ===
df = pd.read_parquet(input_file)

# === 随机抽样 ===
# 如果原数据量小于200，会自动取全部样本
sample_df = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)

# === 重新编号（可选） ===
if "index" in sample_df.columns:
    sample_df["index"] = range(len(sample_df))

# === 保存为新的 parquet 文件 ===
sample_df.to_parquet(output_file, index=False)

print(f"✅ 已从 {input_file} 随机抽取 {len(sample_df)} 条样本，保存至 {output_file}")
