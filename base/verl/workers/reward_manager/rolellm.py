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
import json
import re
import numpy as np
import os
from collections import defaultdict

from verl import DataProto
from verl.utils.reward_score.roleplay import predict
from verl.workers.reward_manager.tools import is_valid_think_format, remove_think_block


RAW_PROMPT = """你是一名知识丰富的模型性能评估专家，你熟悉并了解各种问答任务可能涉及的知识。
现在，你需要协助用户进行模型回复质量的评估，主要考量模型针对问题的回复是否**符合客观事实或应有的表现**。
注意：你应**忽略回复中的风格或个性描述**，而只**聚焦于其最本质的描述内容是否合理，无需纠结具体细节，大意相同或符合逻辑即认为合理**。
用户将会给出模型接受到的问题、输出的回答。
**你的输出应为可被json.load解析的json格式!**
输出共包含两项：reason(解释你给出该打分的理由，应为字符串格式) 以及 score(给出你的打分，该分数应为0-5之间的整数，越大越好)"""

CUS_PROMPT = """你是一名知识丰富的模型性能评估评估专家，你熟悉并了解各种不同角色的背景，设定以及其语言风格。
现在，你需要协助用户进行模型回复质量的评估，主要考量模型针对问题的回复是否**符合角色{role}的语言风格**。
注意：你应**聚焦于其是否遵循了{role}的说话风格以及习惯**，回复越符合角色描述得分越高。
以下是对{role}的描述，你应当参考它们进行打分：{role_desc}
用户将会给出模型接受到的问题、输出的回答。
**你的输出应为可被json.load解析的json格式!**
输出共包含两项：reason(解释你给出该打分的理由，应为字符串格式) 以及 score(给出你的打分，该分数应为0-5之间的整数，越大越好)"""

SPE_PROMPT = """你是一名知识丰富的模型性能评估评估专家，你熟悉并了解各种不同角色的背景，设定以及其所具有的专有知识或记忆。
现在，你需要协助用户进行模型回复质量的评估，主要考量模型针对问题的回复是否**符合角色{role}所具备的知识或记忆**。
注意：你应**聚焦于该回答是否遵循了{role}所了解的知识和经历，不应出现任何角色不了解的事物**，回复涵盖越多与角色相关的知识和记忆得分越高。
以下是对{role}的描述，你应当参考它们进行打分：{role_desc}
用户将会给出模型接受到的问题、输出的回答。
**你的输出应为可被json.load解析的json格式!**
输出共包含两项：reason(解释你给出该打分的理由，应为字符串格式) 以及 score(给出你的打分，该分数应为0-5之间的整数，越大越好)"""

SYSTEM_PROMPT = {
    "raw": RAW_PROMPT,
    "cus": CUS_PROMPT,
    "spe": SPE_PROMPT
}


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


class RolellmRewardManager:
    """The reward manager with pairing logic moved from ray_trainer.
    """

    def __init__(self, tokenizer, num_examine, config=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config = config

    def _build_messages(self, role_name: str, role_desc: str, question: str, model_answer: str, task_id: str):
        """构造评审消息（增强版：支持 truth_answer 为列表）"""
        
        # 1️⃣ 选择合适的系统提示词
        if task_id == "raw":
            system_prompt = SYSTEM_PROMPT[task_id]
        elif task_id in ("spe", "cus"):
            system_prompt = SYSTEM_PROMPT[task_id].format(role=role_name, role_desc=role_desc)
        else:
            system_prompt = (
                "你是一个角色扮演的效果对比助手，你会根据输出的角色特征和质量来对模型进行排名，"
                "然后使用Python dict list输出结果。"
            )


        # 3️⃣ 构造最终 user prompt
        user_prompt = (
            f"接受到的问题：\n{question}\n\n"
            f"输出的回答：\n{model_answer}\n\n"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


    def _judge_model(self, role_name: str, role_desc: str, question: str, model_answer: str, task_name: str, max_try: int = 3, has_thinking = False) -> int:
        """调用vLLM裁判，返回1表示模型胜，0表示负/平/解析失败"""
        import time
        ori_answer = model_answer
        if has_thinking:
            if is_valid_think_format(model_answer):
                model_answer = remove_think_block(model_answer)
            else:
                return {
                        "score": 0,
                        "task_name": task_name,
                        "reason": "模型回答格式错误",
                        "answer": ori_answer
                    }
        
        messages = self._build_messages(role_name, role_desc, question, model_answer, task_name)
            
        for attempt in range(1, max_try + 1):
            try:
                content = predict(messages)
                data = extract_first_json(content)
                
                reason = data.get("reason", "")
                score = float(data.get("score", 0))

                # 写入日志
                record = {
                    "score": score,
                    "task_name": task_name,
                    "reason": reason,
                    "messages": messages,
                    "answer": ori_answer
                }

                return record
    
            except Exception as e:
                print(f"[judge][尝试 {attempt}/{max_try}] 解析失败: {e}")
                if content is not None:
                    print(content)
                if attempt < max_try:
                    time.sleep(1)
                    continue
                else:
                    print("[judge] 多次尝试失败，返回默认分数 0")
                    # 失败也要记录
                    fail_record = {
                        "score": 0,
                        "task_name": task_name,
                        "reason": str(e),
                        "messages": messages,
                        "answer": ori_answer
                    }
                    return fail_record

    def __call__(self, data: DataProto, step=0, has_thinking=False, log_path=None):
        task_scores = defaultdict(list)
        log_list = []

        for i in range(len(data)):
            data_item = data[i]
            
            # 获取基本 token 序列
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            valid_response_ids = data_item.batch['responses'][:valid_response_length]

            model_answer = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # 提取角色与问题信息
            extra_info = data_item.non_tensor_batch['extra_info']
            role_name = extra_info['role_name']
            role_desc = extra_info['role_desc']
            question = extra_info['user_question']
            task_name = extra_info['task_name']

            # 调用裁判模型
            record = self._judge_model(
                role_name=role_name,
                role_desc=role_desc,
                question=question,
                model_answer=model_answer,
                task_name=task_name,
                has_thinking=has_thinking
            )

            score = float(record['score'])
            score = min(max(score, 0.0), 5.0)
            task_scores[task_name].append(score)
            
            # 所有样本都要经过cus评估
            record = self._judge_model(
                role_name=role_name,
                role_desc=role_desc,
                question=question,
                model_answer=model_answer,
                task_name='cus',
                has_thinking=has_thinking
            )

            spe_score = float(record['score'])
            spe_score = min(max(spe_score, 0.0), 5.0)
            task_scores['cus'].append(spe_score)

            log_list.append({
                "score": score,
                "cus_score": spe_score,
                "task_name": task_name,
                "reason": record.get("reason", ""),
                "answer": model_answer,
                "messages": record.get("messages", [])
            })

        for k in list(task_scores.keys()):
            task_scores[k] = np.array(task_scores[k], dtype=float)

        log_dir = os.getenv("LOG_DIR", ".")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"judge_results_{step}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_list, f, ensure_ascii=False, indent=2)
            
        
        return task_scores
