import json
from collections import defaultdict
import random

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[警告] 无法解析行：{line[:100]}...")
    return data

def merge_and_deduplicate(data_list, language):
    merged = defaultdict(lambda: {"role": None, "question": None, "generated": set()})
    for d in data_list:
        role, question = d["role"], d["question"]
        key = (role, question)
        merged[key]["role"] = role
        merged[key]["question"] = question
        merged[key]["generated"].update(d.get("generated", []))
        merged[key]["language"] = language
    # 转换回列表
    merged_list = []
    for v in merged.values():
        v["generated"] = list(v["generated"])
        merged_list.append(v)
    return merged_list

def main(file_a, file_c, output_path):
    # 1. 读取文件
    data_a = load_jsonl(file_a)
    data_c = load_jsonl(file_c)
    print(len(data_a))
    print(len(data_c))

    # 2. 合并 a + b, 然后分别去重
    merged_ab = merge_and_deduplicate(data_a, language = "en")
    merged_c = merge_and_deduplicate(data_c, language = "cn")

    # 3. 按比例抽样（1:1，尽量多）
    n = min(len(merged_ab), len(merged_c))
    n_a = n
    n_c = n
    print(f"将从 A 组抽取 {n_a} 条样本, C 组抽取 {n_c} 条样本 （共 {n_a + n_c} 条）")

    random.shuffle(merged_ab)
    random.shuffle(merged_c)

    selected_ab = merged_ab[:n_a]
    selected_c = merged_c[:n_c]

    combined = selected_ab + selected_c
    random.shuffle(combined)

    # 4. 输出为 jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 输出完成，共 {len(combined)} 条样本 -> {output_path}")

if __name__ == "__main__":
    # 修改为你自己的路径
    a = f"/gemini/space/private/cgn/project/cllm_rl/data/roleplay/ori/spe_en.jsonl"
    c = f"/gemini/space/private/cgn/project/cllm_rl/data/roleplay/ori/spe.jsonl"
    out = f"/gemini/space/private/cgn/project/cllm_rl/data/roleplay/tem/test_cus.jsonl"
    main(a, c, out)
