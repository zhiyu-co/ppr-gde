import pandas as pd

def parquet_diff(file_a, file_b, output_file):

    df_a = pd.read_parquet(file_a)
    df_b = pd.read_parquet(file_b)

    # 提取 B 中所有 (index, user_question)
    keys_b = set(
        row["extra_info"]["index"]
        for _, row in df_b.iterrows()
    )

    mask = []
    for _, row in df_a.iterrows():
        key = row["extra_info"]["index"]
        mask.append(key not in keys_b)

    df_c = df_a[mask]

    df_c.to_parquet(output_file, index=False)

    print("A样本数:", len(df_a))
    print("B样本数:", len(df_b))
    print("A独有样本:", len(df_c))
    
    
    
file_a = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/char_rm_4096.parquet"  
file_b = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/char_rm_2048.parquet"  

output_file = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/char_rm_test_2048.parquet"  


parquet_diff(file_a, file_b, output_file)