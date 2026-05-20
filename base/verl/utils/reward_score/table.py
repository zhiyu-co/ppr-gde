# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import time
import json
import re
import numpy as np
import pandas as pd
import io
import sys
import requests
import random
from collections import defaultdict


# def execute_python_code(code):
#     """执行Python代码并获取最后的输出作为答案"""
#
#     try:
#         # 重定向标准输出
#         old_stdout = sys.stdout
#         sys.stdout = buffer = io.StringIO()
#
#         # 执行代码
#         local_vars = {}
#         exec(code, local_vars)
#
#         # 获取标准输出结果
#         output = buffer.getvalue().strip()
#         sys.stdout = old_stdout
#
#         if output:
#             return output, None
#         else:
#             return None, "执行代码后没有任何输出。"
#
#     except Exception as e:
#         sys.stdout = old_stdout
#         error_message = f"代码执行失败: {e}"
#         return None, error_message


def extract_python_code(prediction):
    """使用正则表达式提取三反引号包裹的Python代码"""
    pattern = r"```python(.*?)```"
    code_blocks = re.findall(pattern, prediction, re.DOTALL)
    if code_blocks:
        code = code_blocks[-1]
    else:
        code = "\n".join(code_blocks)

    return code


def execute_python_code(code):
    st = time.time()
    # python exec API Server
    BASE_URLs = ["http://10.127.23.252:20613", "http://10.127.23.252:29026", "http://10.127.23.252:22484", "http://10.127.23.252:49805"]
    # BASE_URL = "http://10.127.23.252:45735"
    code_data={"code": code}
    max_try = 8
    while max_try>0:
        try:
            BASE_URL = random.choice(BASE_URLs)
            response = requests.post(f"{BASE_URL}/python", json=code_data)
            print("exec python api cost time:{}s".format(time.time() - st))
            return response.json()['result'], response.json()['error_message']
        except Exception as e:
            max_try -= 1
            print("Warning! exec python api error.............\n{}".format(e))
            time.sleep(20)
    error_message = f"调用api失败"
    print("exec python api cost time:{}s".format(time.time()-st))
    return None, error_message


def json_default(obj):
    """自定义 JSON 序列化函数"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def code_format_eval(prediction):
    if prediction.count("```python") != 1:
        return False
    code = extract_python_code(prediction)
    if code and "print(" in code and "pd.read_csv" in code:
        return True
    else:
        return False


def code_exec_result(prediction):
    prediction = extract_python_code(prediction)
    print("\nCleaned code:{}".format(prediction))
    result, error_message = execute_python_code(prediction)
    print("Exec result:{}...".format(str(result)[:500]))
    print("Exec error:{}".format(error_message))
    return result, error_message


def api_reward_model(question, model_output, ref_answer):
    from openai import OpenAI
    st = time.time()

    prompt = "你是一个评判助手，你的任务是根据问题和提供的标准答案来评估其他答案的正确性，判断的标准是，其他答案跟标准答案在关键结果上是否一致，" \
             "如果一致输出1，否则输出0，除此外不要输出其他内容。\n问题：{}\n标准答案：\n{}\n其他答案：\n{}"
    if "</think>" in model_output:
        content = prompt.format(question, ref_answer, model_output.split("</think>")[-1])
    else:
        content = prompt.format(question, ref_answer, model_output)

    openai_api_key = "EMPTY"
    openai_api_base = "http://10.127.23.252:41935/v1"
    # openai_api_key = "sk-telenlp1234"
    # openai_api_base = "http://10.30.129.200:25242/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, )

    def predict(query):
        max_try = 5
        max_tokens = 8192
        while max_try>0:
            try:
                model_name = client.models.list().data[0].id
                response = client.chat.completions.create(
                    model=model_name,
                    messages=query,
                    temperature=0.3,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    extra_body={
                        "repetition_penalty": 1.01,
                        "skip_special_tokens": False,
                        "spaces_between_special_tokens": False
                    },
                )
                max_try = -1
                return response.choices[0].message.model_dump()["content"]
            except:
                max_try -= 1
                time.sleep(60)
                print("Warning! reward api error.............")
        print("***inference failed***")
        return ''

    messages = [{"role": "user", "content": content}]
    answer = predict(messages)
    print("reward model output: {}".format(answer))
    print("reward model api cost time:{}s".format(time.time() - st))
    return answer


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    positions['final_end'] = len(processed_str)-1
    positions['think_length'] = positions['think_end'] - positions['think_start']
    positions['answer_length'] = positions['final_end'] - positions['think_end']

    # Verify tag order
    # if positions['think_start'] > positions['think_end'] or \
    #         (positions['answer_length'] > positions['think_length']):
    if '<|im_end|>' not in processed_str and "<_end>" not in processed_str and "<|endoftext|>" not in processed_str \
            and "<｜end▁of▁sentence｜>" not in processed_str:
        validation_passed = False
    elif positions['think_start'] > positions['think_end']:
        print("  [Error] Incorrect tag order: Expected <think>...</think>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    # check code format
    if validation_passed:
        processed_str = processed_str.split("</think>")[-1]
        validation_passed = code_format_eval(processed_str)

    return validation_passed


def compute_score(solution_str, ground_truth, prompt_str=None, format_reward=1, extra_info=None) -> float:
    """
    get rule based score
    :param solution_str:
    :param ground_truth:
    :param prompt_str:
    :param format_reward:
    :return: Float score
    """
    ori_solution_str = solution_str
    if "<think>\n\n</think>" in prompt_str:
        solution_str = "<think>\n\n</think>" + solution_str
    if "<think>" not in solution_str:
        solution_str = "<think>" + solution_str

    if "<｜Assistant｜>" in prompt_str:
        question = prompt_str.split("输入问题：\n")[1].split("<｜Assistant｜>")[0]
    elif "<_bot>" in prompt_str:
        question = prompt_str.split("输入问题：\n")[1].split("<_bot>")[0]
    elif "<|im_end|>" in prompt_str:
        question = prompt_str.split("输入问题：\n")[1].split("<|im_end|>")[0]
    else:
        question = prompt_str

    # Validate response structure
    format_correct = validate_response_structure(solution_str)
    format_score = format_reward if format_correct else -abs(format_reward)

    print(f"\n[Question]\n{question}")
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    retval = 0
    new_samples = []
    critic_sample, rethink_sample, compare_sample = None, None, None
    code_str = None
    if format_correct:
        try:
            code_str = solution_str.split("</think>")[-1].replace("<|im_end|>", "").replace("<｜end▁of▁sentence｜>", "").replace("<_end>", "").replace("<|endoftext|>", "").strip()
            exec_result, error_message = code_exec_result(code_str)
            if exec_result:
                if '1' in api_reward_model(question, exec_result, ground_truth):
                    retval = 2
                    critic_sample = add_critic_prompt(prompt_str + ori_solution_str, '1')
                else:
                    retval = -1.5
                    rethink_sample = add_rethink_prompt(prompt_str + ori_solution_str, exec_result, error_message,
                                                        ground_truth)
                    critic_sample = add_critic_prompt(prompt_str + ori_solution_str, '0')
            else:
                retval = -1.5
                rethink_sample = add_rethink_prompt(prompt_str + ori_solution_str, exec_result, error_message, ground_truth)
                critic_sample = add_critic_prompt(prompt_str + ori_solution_str, '0')

        except Exception as e:
            print(e)

    total_score = format_score + retval
    if code_str and total_score != -1:
        compare_sample = add_compare_prompt(prompt_str + ori_solution_str, prompt_str, code_str, total_score, extra_info)

    if critic_sample:
        new_samples.append(critic_sample)
    if rethink_sample:
        new_samples.append(rethink_sample)
    if compare_sample:
        new_samples.append(compare_sample)

    return total_score, new_samples


max_rethink_sample_num = 1000
rethink_sample_num = 0


def add_rethink_prompt(content, exec_result, error_message, ground_truth):
    global max_rethink_sample_num, rethink_sample_num
    if '判断你刚才的回答是否正确，如果正确回答1，如果否回答0' in content:
        return None
    if "现在你是一位表格理解任务专家，给你一段表格信息描述，" in content:
        return None
    # 限制反思的次数
    if content.count("code executor") >= 1:
        return None
    if rethink_sample_num > max_rethink_sample_num:
        return None

    if random.randint(0, 32) != 0:
        return None
    if exec_result:
        if len(exec_result)>200:
            content = content + "\n<|im_start|>code executor\n代码执行结果：\n{}......\n根据这个代码执行结果，你好像回答得不对，你再想想<|im_end|>\n<|im_start|>assistant\n<think>".format(exec_result[:200])
        else:
            content = content + "\n<|im_start|>code executor\n代码执行结果：\n{}\n根据这个代码执行结果，你好像回答得不对，你再想想<|im_end|>\n<|im_start|>assistant\n<think>".format(exec_result)
    else:
        content = content + "\n<|im_start|>code executor\n代码执行报错结果：\n{}\n根据这个代码执行结果，你好像回答得不对，你再想想<|im_end|>\n<|im_start|>assistant\n<think>".format(error_message)

    data = {
        "data_source": 'table_rethink',
        "prompt": [{
            "role": "user",
            "content": content
        }],
        "ability": "table_rethink",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth
        },
        "extra_info": {
            'split': 'train',
            'index': None
        }
    }
    # print(data)
    rethink_sample_num += 1
    return data


max_critic_sample_num = 1000
critic_sample_num = 0


def add_critic_prompt(content, ground_truth):
    global max_critic_sample_num, critic_sample_num
    if "<|im_start|>code executor" in content:
        return None
    if critic_sample_num > max_critic_sample_num:
        return None
    critic_prompt = "判断你刚才的回答是否正确，如果正确回答1，如果否回答0，先给出一步一步的分析思考过程，然后把最终答案放进\\boxed{}里面。"
    # 限制critic的轮次
    if critic_prompt in content:
        return None
    if "现在你是一位表格理解任务专家，给你一段表格信息描述，" in content:
        return None
    if random.randint(0, 32) != 0:
        return None
    content += "\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>".format(critic_prompt)

    data = {
        "data_source": 'table_critic',
        "prompt": [{
            "role": "user",
            "content": content
        }],
        "ability": "table",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth
        },
        "extra_info": {
            'split': 'train',
            'index': None
        }
    }
    # print(data)
    critic_sample_num += 1
    return data


max_compare_sample_num, compare_sample_num = 1000, 0
compare_data = defaultdict(list)
compare_data_count = 0


def add_compare_prompt(content, prompt_str, code_solution_str, score, extra_info=None):
    global max_compare_sample_num, compare_sample_num, compare_data, compare_data_count

    if extra_info and 'test' in extra_info['split'].lower():
        return None
    if '判断你刚才的回答是否正确，如果正确回答1，如果否回答0' in content or "<|im_start|>code executor" in content:
        return None
    compare_prompt = "判断上述的答案A和答案B哪个更好，如果答案A更好输出1，如果答案B更好输出0，先给出一步一步的分析思考过程，然后把最终答案放进\\boxed{}里面。"
    # 限制compare的轮次
    if compare_prompt in content:
        return None
    if compare_sample_num > max_compare_sample_num:
        return None

    compare_data[prompt_str].append([prompt_str, code_solution_str, score])
    compare_data_count += 1

    # 等待数据量积累到一定程度
    if compare_data_count < 1000:
        return None

    # 随机抽取pair对
    flag, flag_max = True, 5
    while flag and flag_max>0:
        rand_key = random.choice(list(compare_data.keys()))
        if len(compare_data[rand_key]) >= 2:
            a, b = compare_data[rand_key][-1], compare_data[rand_key][-2]
            if a[-1] != b[-1]:
                flag = False
                compare_data[rand_key].pop()
                compare_data[rand_key].pop()
        flag_max -= 1
    # 抽取不到的情况
    if flag:
        return None

    if random.randint(0, 32) != 0:
        return None

    prompt_str = prompt_str.split('<|im_end|>\n<|im_start|>assistant')[0]
    prompt_str = "以下是提供的表格信息：" + prompt_str.split("以下是提供的表格信息：")[1]
    prompt_str = prompt_str.split("确保最终答案是 Python 代码的最后一行，并且只能")[0] + "输入问题：" + prompt_str.split("输入问题：")[1]

    prompt_str = "现在你是一位表格理解任务专家，给你一段表格信息描述，一个问题，以及该问题的A、B两个用python代码解题的答案，你的任务是判断出哪个答案更正确，"+ prompt_str
    content = "{}\n答案A：\n{}\n答案B:\n{}\n{}<|im_end|>\n<|im_start|>assistant\n<think>".format(prompt_str, a[1], b[1], compare_prompt)

    data = {
        "data_source": 'table_compare',
        "prompt": [{
            "role": "user",
            "content": content
        }],
        "ability": "table",
        "reward_model": {
            "style": "rule",
            "ground_truth": '1' if a[-1]>b[-1] else '0'
        },
        "extra_info": {
            'split': 'train',
            'index': None
        }
    }

    # print(data)
    compare_sample_num += 1
    return data

