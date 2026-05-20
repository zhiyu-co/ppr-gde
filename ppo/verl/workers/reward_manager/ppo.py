# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

import torch
import re
from typing import Optional, Union

from verl import DataProto
from verl.utils.reward_score.roleplay import predict
from verl.workers.reward_manager.tools import is_valid_think_format, remove_think_block


SYS_EVALUATION_CN = """
你是一位非常严谨且专业的文本评估专家，你将协助用户根据给定的对话内容和评估标准对回答进行质量评估。
"""

USR_PROMPT_CN = """
**严格判断标准：**
1. 角色回复的语言风格和表达方式必须完全符合角色描述，遵循用户的指令和满足用户需求
2. 回复的语气、措辞必须符合该角色的原始性格和说话方式，符合就角色身份
3. 不得回复不安全的内容，不安全包括：违法犯罪、色情低俗、不道德、歧视偏见、泄露隐私等
4. 回答需要符合人物的性格特征，内容不能过长，不能冗余，需要精简且高质量的回复

角色名称：【{role_name}】
角色描述：【{role_desc}】
用户问题：【{user_question}】

角色的最终回答：
{response}

你的回答必须为 [[score]] 其中score是 1-5 之间的任一整数，数值越大说明角色回答越符合判断标准。
**不要包含任何额外文字。**
"""

SYS_EVALUATION_EN = """
You are a highly rigorous and professional text evaluation expert. You will assist the user in assessing the quality of a response according to the given dialogue content and evaluation criteria.
"""

USR_PROMPT_EN = """
**Strict Evaluation Criteria:**
1. The language style and manner of expression in the role's response must fully match the role description, follow the user's instructions, and satisfy the user's needs.
2. The response must remain consistent throughout, avoid deviating from the role's original personality and discourse style, and stay in character.
3. The response must not contain unsafe content. Unsafe content includes, but is not limited to: illegal or criminal acts, vulgarity, emotional manipulation, immoral behavior, discriminatory or biased remarks, privacy leakage, etc.
4. The response should reflect the character's personality traits. It should not be overly long or verbose, and should instead be concise, high-quality, and to the point.

Role Name: [{role_name}]
Role Description: [{role_desc}]
User Question: [{user_question}]

Final Response of the Role:
{response}

Your reply must be in the format [[score]], where score is an integer between 1 and 5. A higher value indicates that the role response better satisfies the evaluation criteria.
**Do not include any additional text.**
"""



def extract_score(text: str) -> Optional[Union[int, float]]:
    """
    从字符串中提取形如 [[score]] 的分数。
    """
    pattern = r"\[\[\s*([-+]?\d+(?:\.\d+)?)\s*\]\]"
    match = re.search(pattern, text)
    
    if not match:
        return None
    
    score_str = match.group(1)
    
    return int(score_str)


class PPORewardManager:
    """The reward manager with pairing logic moved from ray_trainer.
    """

    def __init__(self, tokenizer, config=None, p_function=None):
        self.tokenizer = tokenizer
        self.config = config
        self.print = print if p_function is None else p_function
        
    def _build_messages(self, role_name: str, role_desc: str, question: str, model_answer: str, language: str):
        """构造评审消息（增强版：支持 truth_answer 为列表）"""
        
        if language == "cn":
            system_prompt = SYS_EVALUATION_CN
            user_prompt = USR_PROMPT_CN.format(role_name=role_name, role_desc=role_desc, user_question=question, response=model_answer)
        elif language == "en":
            system_prompt = SYS_EVALUATION_EN
            user_prompt = USR_PROMPT_EN.format(role_name=role_name, role_desc=role_desc, user_question=question, response=model_answer)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


    def _judge_model(self, role_name: str, role_desc: str, question: str, model_answer: str, language: str, max_try: int = 3) -> int:
        """调用vLLM裁判，返回1表示模型胜，0表示负/平/解析失败"""
        import time
        messages = self._build_messages(role_name, role_desc, question, model_answer, language)
            
        for attempt in range(1, max_try + 1):
            try:
                content = predict(messages)
                score = extract_score(content)

                return score
    
            except Exception as e:
                print(f"[judge][尝试 {attempt}/{max_try}] 解析失败: {e}")
                if content is not None:
                    print(content)
                if attempt < max_try:
                    time.sleep(1)
                    continue
                else:
                    print("[judge] 多次尝试失败，返回默认分数 0")
                    return 0

    def __call__(self, data: DataProto, has_thinking=False, log_path=None):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        thinking_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        diversity_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        for i in range(len(data)):
            data_item = data[i]
            
            # 获取基本 token 序列
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()

            valid_response_ids = data_item.batch['responses'][:valid_response_length]

            model_answer = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # 提取角色与问题信息
            extra_info = data_item.non_tensor_batch['extra_info']
            role_name = extra_info['role_name']
            role_desc = extra_info['role_desc']
            question = extra_info['user_question']
            task_name = extra_info['task_name']
            language = extra_info['language']

            # 调用裁判模型
            score = self._judge_model(
                role_name=role_name,
                role_desc=role_desc,
                question=question,
                model_answer=model_answer,
                language=language
            )
            score = (min(max(float(score), 0.0), 5.0) - 1) / 4
            
            reward_tensor[i, valid_prompt_length - 1] = score
            
        return reward_tensor, thinking_tensor, diversity_tensor