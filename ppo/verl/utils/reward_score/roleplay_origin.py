# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE/2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward function for roleplay training

This module provides reward calculation for roleplay tasks, evaluating:
1. Role consistency (staying in character)
2. Response quality and relevance
3. Language style appropriateness
4. Professional knowledge accuracy

The reward function works with paired responses (rollout=16 generates 8 pairs)
"""

import re
import os
from typing import Tuple, Dict, Any, List

# 使用 OpenAI 兼容客户端调用你部署的 vLLM 服务
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def extract_role_info(prompt_str: str) -> Dict[str, str]:
    """
    从提示词中提取角色信息,感觉没啥用
    
    Args:
        prompt_str: 完整的提示词字符串
    
    Returns:
        包含角色信息的字典
    """
    role_info = {}
    
    # 提取角色名称
    role_match = re.search(r'你的名字是：(.+?)\n', prompt_str)
    if role_match:
        role_info['role_name'] = role_match.group(1).strip()
    
    # 提取角色描述
    desc_match = re.search(r'你的背景和描述是：(.+?)\n', prompt_str)
    if desc_match:
        role_info['description'] = desc_match.group(1).strip()
    
    # 提取用户问题
    question_match = re.search(r'用户问题：(.+?)(?:\n|$)', prompt_str)
    if question_match:
        role_info['user_question'] = question_match.group(1).strip()
    
    return role_info


def predict(messages):
    """调用部署的 vLLM(OpenAI 兼容) 服务生成响应。

    优先从环境变量读取：
      - VLLM_API_BASE: 例如 http://127.0.0.1:8355/v1
      - VLLM_API_KEY: 例如 EMPTY 或你的鉴权KEY
    如未设置，则回落到用户给定的默认值。
    """
    api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8355/v1")
    api_key  = os.environ.get("VLLM_API_KEY", "EMPTY")

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
                temperature=0.7,
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


def compare_responses(response_a: str, response_b: str, first_category: str, 
                     role_info: Dict[str, str]) -> Tuple[float, float]:
    """
    使用GSB方式比较两个回复的质量
    
    Args:
        response_a: 回复A
        response_b: 回复B
        first_category: 智能体类别
        role_info: 角色信息
    
    Returns:
        (回复A的分数, 回复B的分数)
    """
    # 定义评判标准
#     evaluation_criteria = """
# **严格判断标准：**
# 1. 智能体回复的语言风格和表达方式必须完全符合智能体的设定，遵循用户的指令和满足用户需求，并且知识准确，对用户有帮助
# 2. Tools类智能体，应该条理清晰，具有专业性
# 3. Education类智能体必须体现教育者的专业素养和权威性，回复要有逻辑层次，体现教学方法，语调应该耐心、引导性强     
# 4. Writing类智能体必须体现写作专家的文学素养和创作能力，语言要富有文采
# 5. IP类智能体：语言表达要符合该IP角色的原始性格和说话方式
# 6. Medical类智能体必须体现医疗专业人员的严谨性和责任感，使用准确的医学术语，要客观、科学，否则业视为不合格
# 7. Government类智能体：语言要正式严谨，符合政务场景的专业表达 
# 8. Life类智能体：语言要更亲切实用，贴近日常生活场景
# 9. Companion类智能体语言要温暖、有同理心，但不能过度情感化，体现陪伴者的支持性和理解性
# 10. 对于特定领域（如政务、医疗、法律、金融）的智能体，其语言风格应该更接近该领域内专家，如果不够专业视为不合格
# 11. 回复的语气、措辞、专业程度必须与智能体身份高度一致
# 12. 即使内容正确，如果语言风格不够贴合角色设定，也视为不合格
# 13. 不得回复不安全的内容，不安全包括：违法犯罪、色情低俗、不道德、歧视偏见、泄露隐私等
# """
    evaluation_criteria = """
**严格判断标准：**
1. 角色回复的语言风格和表达方式必须完全符合角色的设定，遵循用户的指令和满足用户需求，并且知识准确，对用户有帮助
2. Character类角色：语言表达要符合该IP角色的原始性格和说话方式
3. 对于特定领域（如政务、医疗、法律、金融）的角色，其语言风格应该更接近该领域内专家，如果不够专业视为不合格
4. 回复的语气、措辞、专业程度必须与角色身份高度一致
5. 即使内容正确，如果语言风格不够贴合角色设定，也视为不合格
6. 不得回复不安全的内容，不安全包括：违法犯罪、色情低俗、不道德、歧视偏见、泄露隐私等
"""

    # 第一次调用：回答A在前，回答B在后
    judging_user_prompt_1 = f"""{evaluation_criteria}
角色类型：【{first_category}】
最后角色的回答，
回答A：
{response_a}
回答B：
{response_b}
请从回答A和回答B中选择更好的一个。你的回答必须以 "the better answer is: A" 或 "the better answer is: B" 开始，不包含任何额外文字。
"""

    judging_system_prompt = """You are a highly stern and professional expert in evaluating text, responsible for assessing the quality of responses based on the given dialogue content and evaluation criteria."""

    query_messages_1 = [
        {"role": "system", "content": judging_system_prompt},
        {"role": "user", "content": judging_user_prompt_1}
    ]

    # 第二次调用：回答B在前，回答A在后
    judging_user_prompt_2 = f"""{evaluation_criteria}
智能体类型：【{first_category}】
最后智能体的回答，
回答A：
{response_b}
回答B：
{response_a}
请从回答A和回答B中选择更好的一个。你的回答必须以 "the better answer is: A" 或 "the better answer is: B" 开始，不包含任何额外文字。
"""

    query_messages_2 = [
        {"role": "system", "content": judging_system_prompt},
        {"role": "user", "content": judging_user_prompt_2}
    ]

    try:
        # 第一次调用VLLM API
        vllm_response_1 = predict(query_messages_1)
        # 第二次调用VLLM API
        vllm_response_2 = predict(query_messages_2)
        
        # 解析第一次响应
        match_a_1 = re.search(r"the better answer is:\s*A", vllm_response_1, re.IGNORECASE)
        match_b_1 = re.search(r"the better answer is:\s*B", vllm_response_1, re.IGNORECASE)
        
        # 解析第二次响应
        match_a_2 = re.search(r"the better answer is:\s*A", vllm_response_2, re.IGNORECASE)
        match_b_2 = re.search(r"the better answer is:\s*B", vllm_response_2, re.IGNORECASE)
        
        # 根据GSB规则计算分数
        # 如果两次都是A更好，那么A得1分
        # 如果两次都是B更好，那么A得-1分
        # 其他情况A得0分
        
        if match_a_1 and match_b_2:
            # 两次都是A更好
            score_a = 1.0
            score_b = -1.0
        elif match_b_1 and match_a_2:
            # 两次都是B更好
            score_a = -1.0
            score_b = 1.0
        else:
            # 其他情况（不一致或无法判断）
            score_a = 0.0
            score_b = 0.0
            
    except Exception as e:
        print(f"[WARNING] VLLM API call failed: {e}")
    
    return score_a, score_b


def compute_score(data_source: str, solution_str: str, ground_truth: str, 
                  extra_info: Dict[str, Any] = None) -> Tuple[float, List]:
    """
    计算roleplay任务的奖励分数
    
    Args:
        data_source: 数据来源
        solution_str: 当前回复
        ground_truth: 标准答案（在这个场景下不使用）
        extra_info: 额外信息，包含配对回复和角色信息
    
    Returns:
        (奖励分数, 新样本列表)
    """
    if not extra_info:
        return 0.0, []
    
    # 从extra_info中提取信息
    paired_response = extra_info.get('paired_response', '')
    role_name = extra_info.get('role_name', '')
    first_category = extra_info.get('first_category', '')
    prompt_str = extra_info.get('prompt_str', '')
    
    # if not paired_response or not prompt_str:
    #     return 0.0, []
    
    # 提取角色信息
    role_info = extract_role_info(prompt_str)
    
    # 比较两个回复
    score_current, score_paired = compare_responses(
        solution_str, 
        paired_response, 
        first_category, 
        role_info
    )
    
    # 返回当前回复的分数和新样本列表
    return score_current, []
