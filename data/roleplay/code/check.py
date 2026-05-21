import re, json
from openai import OpenAI


def extract_first_json(text: str):
    """
    从文本中提取第一个完整 JSON 对象。
    """

    # 去掉 Markdown 包裹
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")

    # 正则匹配 { ... } （非贪婪）
    pattern = r"\{[\s\S]*?\}"
    matches = re.findall(pattern, text)

    if not matches:
        raise ValueError("未找到 JSON 对象")

    # 只解析第一个
    json_str = matches[0]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败：{e}\nJSON内容：\n{json_str}")

SYS = """
你是一名知识丰富的语言专家，熟悉各种角色的设定与背景知识，你需要协助用户进行一些数据处理工作。
用户将会输入角色名，角色描述以及问题，而你需要判断该角色是否具备回答该问题的能力或所需知识。
也就是说，你需要判断使用这样一个问题来询问对应角色是否合适，不会产生该角色无法回答的情况。
你接受到的输入将如下所示：

角色名：
角色描述：
问题：

而你需要以json格式返回你的回答，如下格式所示：
{{
    "reason": 你做出判断的理由，以字符串形式表现,
    "res": 该问题是否合适，以字符串形式表现，仅可以输出 yes 或者 no
}}
"""

USR = """
角色名：{role_name}
角色描述：{role_desc}
问题：{question}
"""

def build_messages(role_name, role_desc, question):
    return [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USR.format(role_name=role_name, role_desc=role_desc, question=question)}
    ]

def predict(messages):
    """调用部署的 vLLM(OpenAI 兼容) 服务生成响应。

    优先从环境变量读取：
      - VLLM_API_BASE: 例如 http://10.244.78.132:8010/v1
      - VLLM_API_KEY: 例如 EMPTY 或你的鉴权KEY
    如未设置，则回落到用户给定的默认值。
    """
    # api_base = os.environ.get("VLLM_API_BASE", "http://10.244.69.34:8355/v1") 
    api_base = "http://10.244.23.160:8355/v1"
    api_key  = "EMPTY"

    if OpenAI is None:
        raise RuntimeError("openai SDK 未安装，无法调用兼容接口。请先 pip install openai>=1.0.0")

    client = None
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
    except Exception as e:
        raise RuntimeError(f"初始化 OpenAI 客户端失败: {e}")

    max_try = 5
    last_err = None
    while max_try > 0:
        try:
            # 获取服务端可用模型
            model_name = client.models.list().data[0].id

            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.5,
                top_p=0.9,
                max_tokens=2048,
                extra_body={
                    "repetition_penalty": 1.05,
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            max_try -= 1
            print(f"[ERROR] predict调用失败，重试中... 剩余{max_try}次。错误: {e}")
    raise RuntimeError(f"predict多次重试失败: {last_err}")


def judge_model(role_name: str, role_desc: str, question: str, max_try: int = 3):
    """调用vLLM裁判，返回1表示模型胜，0表示负/平/解析失败"""
    import time
    
    messages = build_messages(role_name, role_desc, question)
        
    for attempt in range(1, max_try + 1):
        try:
            content = predict(messages)
            data = extract_first_json(content)
            
            reason = data.get("reason")
            res = data.get("res")
            
            if "yes" in res:
                return {
                    "reason": reason,
                    "res": True
                }
            if "no" in res:
                return {
                    "reason": reason,
                    "res": False
                }
            raise ValueError("返回不合规")

        except Exception as e:
            print(f"[judge][尝试 {attempt}/{max_try}] 解析失败: {e}")
            if content is not None:
                print(content)
            if attempt < max_try:
                time.sleep(1)
                continue
            else:
                print("[judge] 多次尝试失败，返回默认分数 0")
                return {
                    "reason": "error",
                    "res": False
                }
                
                
                
import json
import os
from tqdm import tqdm
from typing import Dict, Optional


def get_last_input_index(output_jsonl_path: str) -> Optional[int]:
    """
    从输出 jsonl 的最后一行读取 _input_index
    """
    if not os.path.exists(output_jsonl_path):
        return None

    last_line = None
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            last_line = line

    if last_line is None:
        return None

    try:
        obj = json.loads(last_line)
        return obj.get("_input_index")
    except Exception:
        return None


def filter_jsonl_with_judge_resume_strict(input_jsonl_path: str, output_jsonl_path: str):
    """
    ✔ tqdm 进度条
    ✔ 严格断点重续（基于输入 index，而非输出行数）
    ✔ 支持删除样本
    """

    # 1. 输入文件总行数（给 tqdm 用）
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # 2. 从输出文件中恢复 input_index
    last_input_index = get_last_input_index(output_jsonl_path)
    start_index = last_input_index + 1 if last_input_index is not None else 0

    print(f"[Resume] total input lines : {total_lines}")
    print(f"[Resume] start from index  : {start_index}")

    with open(input_jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_jsonl_path, "a", encoding="utf-8") as fout:

        pbar = tqdm(
            enumerate(fin),
            total=total_lines,
            initial=start_index,
            desc="Judging samples",
        )

        for idx, line in pbar:
            # 跳过已经处理过的输入
            if idx < start_index:
                continue

            sample = json.loads(line)

            role = sample.get("role")
            desc = sample.get("desc")
            question = sample.get("question")

            if role is None or desc is None or question is None:
                continue

            try:
                judge_res: Dict = judge_model(
                    role_name=role,
                    role_desc=desc,
                    question=question,
                )
            except Exception as e:
                print(f"[Judge Error] idx={idx}: {e}")
                continue

            if judge_res.get("res", False) is True:
                sample["reason"] = judge_res.get("reason", "")
                sample["_input_index"] = idx  # ⭐ 断点重续关键字段
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                fout.flush()
                
input = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/tem/test_raw.jsonl"
output = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/tem/test_raw_new.jsonl"
filter_jsonl_with_judge_resume_strict(input, output)