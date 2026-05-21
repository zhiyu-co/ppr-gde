import json
import random
from collections import defaultdict

def load_jsonl(path):
    """读取 jsonl 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[警告] 无法解析行：{line[:80]}...")
    return data

def load_desc_json(path):
    """读取 JSON 格式的描述文件，返回值列表"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 格式错误：期望是一个对象（键值对），但得到 {type(data)}")
    return list(data.keys()), data

def merge_and_deduplicate(data_list, language):
    """按 instruction 聚合 target 去重"""
    merged = defaultdict(lambda: {"question": None, "target": set()})
    for d in data_list:
        instr = d["question"]
        key = instr
        merged[key]["question"] = instr
        merged[key]["language"] = language
    merged_list = []
    for v in merged.values():
        merged_list.append(v)
    return merged_list

def main(file_a, file_c, ab_desc_file, c_desc_file, output_path):
    random.seed(42)

    # 1. 读取数据
    data_a = load_jsonl(file_a)
    data_c = load_jsonl(file_c)

    # 2. 读取描述（JSON格式）
    ab_roles, hash_en = load_desc_json(ab_desc_file)
    c_roles, hash_cn = load_desc_json(c_desc_file)

    if not ab_roles or not c_roles:
        raise ValueError("描述文件为空，无法分配 role。")

    # 3. 合并并去重
    merged_ab = merge_and_deduplicate(data_a, language = "en")
    merged_c = merge_and_deduplicate(data_c, language = "cn")

    # 4. 1:1 抽样
    n = min(len(merged_ab), len(merged_c))
    print(f"将从 AB 组与 C 组各抽取 {n} 条样本（共 {2*n} 条）")

    random.shuffle(merged_ab)
    random.shuffle(merged_c)

    selected_ab = merged_ab[:n]
    selected_c = merged_c[:n]

    # 5. 随机分配 role
    for item in selected_ab:
        item["role"] = random.choice(ab_roles)
        role = item["role"]
        while role == item["role"]:
            role = random.choice(ab_roles)
        item["role_1"] = role
    for item in selected_c:
        item["role"] = random.choice(c_roles)
        role = item["role"]
        while role == item["role"]:
            role = random.choice(c_roles)
        item["role_1"] = role

    combined = selected_ab + selected_c
    random.shuffle(combined)

    # 6. 输出
    with open(output_path, "w", encoding="utf-8") as f:
        for item in combined:
            out = {
                "role": item["role"],
                "question": item["question"],
                "language": item["language"],
                "desc": hash_cn[item["role"]] if item["language"]=='cn' else hash_en[item["role"]]
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            out = {
                "role": item["role_1"],
                "question": item["question"],
                "language": item["language"],
                "desc": hash_cn[item["role_1"]] if item["language"]=='cn' else hash_en[item["role_1"]]
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"✅ 输出完成，共 {len(combined)} 条样本 -> {output_path}")

if __name__ == "__main__":
    a = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/ori/test_raw_en.jsonl"
    c = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/ori/test_raw_cn.jsonl"
    out = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/tem/test_raw.jsonl"
    ab_desc = "/gemini/space/private/cgn/project/cllm_rl/data/data_set/ZenMoore_RoleBench/profiles-eng/desc.json"
    c_desc = "/gemini/space/private/cgn/project/cllm_rl/data/data_set/ZenMoore_RoleBench/profiles-zh/desc.json"
    main(a, c, ab_desc, c_desc, out)