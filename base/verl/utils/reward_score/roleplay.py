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

from config import VLLM_API_BASE

# 使用 OpenAI 兼容客户端调用你部署的 vLLM 服务
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def extract_role_info(prompt_str: str) -> Dict[str, str]:
    """
    从提示词中提取角色信息
    
    Args:
        prompt_str: 完整的提示词字符串
    
    Returns:
        包含角色信息的字典
    """
    role_info = {}
    
    # 提取角色名称 - 支持多种格式
    role_patterns = [
        r'你是(.+?)，',  # 你是孙悟空，
        r'你是：(.+?)\n',  # 你是：孙悟空
        r'你是([^，。\n]+)',  # 你是孙悟空
    ]
    
    for pattern in role_patterns:
        role_match = re.search(pattern, prompt_str)
        if role_match:
            role_info['role_name'] = role_match.group(1).strip()
            break
    
    # 提取角色描述 - 支持多种格式
    desc_patterns = [
        r'你的特征描述是：(.+?)。',  # 你的特征描述是：xxx。
        r'你的特征描述是：(.+?)\n',  # 你的特征描述是：xxx
        r'特征描述是：(.+?)。',  # 特征描述是：xxx。
        r'特征描述是：(.+?)\n',  # 特征描述是：xxx
    ]
    
    for pattern in desc_patterns:
        desc_match = re.search(pattern, prompt_str)
        if desc_match:
            role_info['description'] = desc_match.group(1).strip()
            break
    
    # 提取用户问题 - 支持多种格式
    question_patterns = [
        r'我的问题是：(.+?)(?:\n|$)',  # 我的问题是：xxx
        r'用户问题：(.+?)(?:\n|$)',  # 用户问题：xxx
        r'问题是：(.+?)(?:\n|$)',  # 问题是：xxx
        r'我想问：(.+?)(?:\n|$)',  # 我想问：xxx
        r'My question is: (.+?)(?:\n|$)'
    ]
    
    for pattern in question_patterns:
        question_match = re.search(pattern, prompt_str)
        if question_match:
            role_info['user_question'] = question_match.group(1).strip()
            break
     
    return role_info


def predict(messages):
    """调用部署的 vLLM(OpenAI 兼容) 服务生成响应。

    优先从环境变量读取：
      - VLLM_API_BASE: 例如 http://127.0.0.1:8355/v1
      - VLLM_API_KEY: 例如 EMPTY 或你的鉴权KEY
    如未设置，则回落到用户给定的默认值。
    """
    # api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8355/v1") 
    api_base = VLLM_API_BASE
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


def compare_responses(response_a: str, response_b: str, first_category: str, role_info: Dict[str, str], language = "cn") -> Tuple[float, float]:
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
    from verl.utils.prompt import EVALUATION_CN, EVALUATION_EN, SYS_EVALUATION_CN, SYS_EVALUATION_EN
    
    # 从role_info获取更多上下文
    """
    role_name = role_info.get('role_name', '') if role_info else ''
    role_desc = role_info.get('description', '') if role_info else ''
    user_question = role_info.get('user_question', '') if role_info else ''
    """

    role_name = role_info.get('role_name', '')
    role_desc = role_info.get('description', '')
    user_question = role_info.get('user_question', '')

    # 第一次调用：回答A在前，回答B在后
    if language == "cn":
        judging_system_prompt = SYS_EVALUATION_CN
        user = EVALUATION_CN
    else:
        judging_system_prompt = SYS_EVALUATION_EN
        user = EVALUATION_EN
    
    judging_user_prompt_1 = user.format(
        role_name=role_name,
        role_desc=role_desc,
        user_question=user_question,
        response_a=response_a,
        response_b=response_b
    ) 

    query_messages_1 = [
        {"role": "system", "content": judging_system_prompt},
        {"role": "user", "content": judging_user_prompt_1}
    ]

    # 第二次调用：回答B在前，回答A在后
    judging_user_prompt_2 = user.format(
        role_name=role_name,
        role_desc=role_desc,
        user_question=user_question,
        response_a=response_b,
        response_b=response_a
    ) 

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
