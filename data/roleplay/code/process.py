import json

def process_question(question: str, language: str) -> str:
    if not isinstance(question, str):
        return question
    
    if language == "cn":
        idx = question.find('，')
        # 删除第一个逗号及之前的内容
        return question[idx + 1 :]

    elif language == "en":
        # 删除逗号及之前内容
        idx = question.find(',')
        new_q = question[idx + 1 :]

        # 删除逗号后紧跟的一个空格（如果有）
        if new_q.startswith(" "):
            new_q = new_q[1:]

        # 将第一个字母大写
        if len(new_q) > 0:
            new_q = new_q[0].upper() + new_q[1:]

        return new_q

    else:
        # 其他 language，不处理
        return question


def process_jsonl(
    input_path: str,
    output_path: str,
    question_key: str = "question",
    language_key: str = "language"
):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Line {line_idx} JSON decode failed, skip")
                continue

            question = sample.get(question_key)
            language = sample.get(language_key)

            if question is not None and language is not None:
                sample[question_key] = process_question(question, language)

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")


import json
from opencc import OpenCC

cc = OpenCC("t2s")  # 繁体 -> 简体


def process_and_merge_jsonl(
    input_path_1: str,
    output_path: str,
    keep_keys=("role", "question", "language")
):
    with open(output_path, "w", encoding="utf-8") as fout:
        with open(input_path_1, "r", encoding="utf-8") as fin:
            for line_idx, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] {input_path_1}:{line_idx} JSON decode failed, skip")
                    continue

                # 只保留指定字段
                new_sample = {k: sample.get(k) for k in keep_keys}

                question = new_sample.get("question")
                language = new_sample.get("language")

                # 中文统一转简体
                if language == "cn" and isinstance(question, str):
                    new_sample["question"] = cc.convert(question)

                fout.write(json.dumps(new_sample, ensure_ascii=False) + "\n")



import json

def dedup_jsonl_by_question(input_path, output_path):
    seen_questions = set()
    kept_samples = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            sample = json.loads(line)
            question = sample.get("question")

            # 如果没有 question 字段，直接跳过或保留（可按需调整）
            if question is None:
                continue

            if question not in seen_questions:
                seen_questions.add(question)
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept_samples += 1

    print(f"去重完成，保留样本数：{kept_samples}")


input = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/tem/test_raw.jsonl"
input_2 = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/tem/test_cus.jsonl"
output = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/tem/test_raw_1.jsonl"

process_and_merge_jsonl(input, output)
# dedup_jsonl_by_question(input, output)
# process_jsonl(input, output)