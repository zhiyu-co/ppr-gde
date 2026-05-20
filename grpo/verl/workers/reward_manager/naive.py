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

import re
from verl import DataProto
# from verl.utils.reward_score import _default_compute_score
import torch
import numpy as np
from collections import defaultdict
from verl.utils.reward_score.roleplay import compute_score, extract_role_info, predict


class NaiveRewardManager:
    """The reward manager with pairing logic moved from ray_trainer.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, config=None, p_function=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # 如果没有传入compute_score，创建一个默认的
        self.compute_score = compute_score or self._create_default_compute_score()
        self.config = config
        self.expected_responses = self.get_expected_responses()  # 新增参数
        self.roleplay_baseline = self._load_roleplay_baseline()
        self.print = print if p_function is None else p_function

    def _load_roleplay_baseline(self):
        """从环境变量读取baseline(jsonl)，支持多个文件，构建 (role, question)->answer 映射（兼容旧接口）"""
        import os, json
        def _load_one(path):
            mp = {}
            if not path:
                return mp
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            role = str(obj.get('role', '')).strip()
                            question = str(obj.get('question', '')).strip()
                            generated = obj.get('generated', [])
                            answer = ''
                            if isinstance(generated, list) and len(generated) > 0:
                                answer = str(generated[0]).strip()
                            elif isinstance(generated, str):
                                answer = generated.strip()
                            if role and question and answer:
                                mp[(role, question)] = answer
                        except Exception:
                            continue
            except Exception as e:
                print(f"[roleplay_baseline] load failed from {path}: {e}")
            print(f"[roleplay_baseline] loaded {len(mp)} entries from {path}")
            return mp

        path1 = os.environ.get('ROLEPLAY_BASELINE_JSONL', '')
        path2 = os.environ.get('BASEMODEL_BASELINE_JSONL', '')

        b1 = _load_one(path1)
        b2 = _load_one(path2)

        # 兼容旧字段（训练不依赖baseline，保持一个默认）
        self.roleplay_baseline1 = b1
        self.roleplay_baseline2 = b2
        merged = {}
        merged.update(b1)
        for k, v in b2.items():
            if k not in merged:
                merged[k] = v
        print(f"[roleplay_baseline] merged total (for fallback): {len(merged)} entries")
        return merged

    def _build_rank_messages(self, role_name: str, role_desc: str, question: str, model_answer: str, baseline_answer: str):
        """构造评审消息"""
        import os
        system_prompt = os.environ.get(
            'EVAL_SYSTEM_PROMPT',
            (
                "你是一个角色扮演的效果对比助手，你会根据输出的角色特征和质量来对模型进行排名，"
                "然后使用Python dict list输出结果。"
            )
        )
        # 两条原则与输出格式，尽量贴合图片规则
        user_prompt = (
            f"下列模型要考察的角色是“{role_name}”，{role_name}的角色描述是“{role_desc}”。你需要根据下面两个原则对下列模型进行排名：\n"
            f"1. 哪一个的角色说话风格更更加明显，说话更加符合角色描述，说话更有特色就越好；\n"
            f"2. 哪一个的答案涵盖了更多与角色相关的知识和记忆，越丰富越好（如果问题中包含了答案，另外角色的相关的知识记忆以参考答案为准）。\n\n"
            f"输入的问题是：\n{question}\n\n"
            f"两个模型对该问题的回答分别为：\n"
            f"[{{\"model\": \"model\", \"answer\": \"{model_answer}\"}}, {{\"model\": \"baseline\", \"answer\": \"{baseline_answer}\"}}]\n\n"
            f"现在请根据上述两个原则，对两个模型进行排名。避免任何位置偏见，并确保模型回答的呈现顺序不会影响你的决定。"
            f"不要对模型给分。最后使用一个包含模型与其排名与这样排名的合理性的列表返回结果，也就是说，请务必使用如下格式返回：\n"
            f"[{{\"model\": <model-name>, \"reason\": <rank-reason>, \"rank\": <model-rank>}}, {{\"model\": <model-name>, \"reason\": <rank-reason>, \"rank\": <model-rank>}}]"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


    def _judge_model_vs_baseline(self, role_name: str, role_desc: str, question: str, model_answer: str, baseline_answer: str) -> int:
        """调用vLLM裁判，返回1表示模型胜，0表示负/平/解析失败"""
        try:
            messages = self._build_rank_messages(role_name, role_desc, question, model_answer, baseline_answer)
            content = predict(messages)
            # print(f"messages: {messages}")
            # print(f"content: {content}")
            import re, json

            # 先用鲁棒正则直接抽取 rank=1 的对应模型名（兼容中英文引号、单双引号、键值顺序）
            q = r'["“”\']'
            pattern = re.compile(
                rf'\{{\s*{q}model{q}\s*:\s*{q}(model|baseline){q}[\s\S]*?{q}rank{q}\s*:\s*(\d+)\s*\}}',
                re.IGNORECASE
            )
            winner = None
            for name, rank in pattern.findall(content):
                if rank.strip() == "1":
                    winner = name.lower()
                    break

            # 回退：尝试解析完整 JSON 数组（不做引号替换，避免破坏字符串）
            if winner is None:
                match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", content)
                if match:
                    try:
                        arr = json.loads(match.group(0))
                        for item in arr:
                            if int(item.get('rank', 0)) == 1:
                                winner = str(item.get('model', '')).lower()
                                break
                    except Exception as e:
                        print(f"[judge] json parse failed: {e}")

            if winner is None:
                return 0
            if 'model' in winner:
                return 1
            if 'baseline' in winner:
                return 0
            return 0
        except Exception as e:
            print(f"[judge] error: {e}")
            return 0

    def mark_dataset(self, train_dataloader):
        self.train_dataloader = train_dataloader

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # 统一走 compute_score 回调；不同数据源内部会再做签名适配
            score, rethink_sample = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                prompt_str=prompt_str,
                extra_info=extra_info,
            )
            scores.append(score)
        # 将精度/得分张量缓存回 data.batch['acc']，便于后续使用
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def group_responses_by_uid(self, data: DataProto):
        """
        根据uid字段对responses进行分组
        
        Args:
            data: DataProto对象
            
        Returns:
            dict: uid -> [response_indices] 的映射
            {'uuid-1': [0,5,12,18,...], 'uuid-2': [3,8,15,22,...], ...}
        """
        uid_to_indices = defaultdict(list)
        
        for i in range(len(data)):
            uid = data.non_tensor_batch['uid'][i]
            uid_to_indices[uid].append(i)
            
        return uid_to_indices

    def pair_responses_for_roleplay(self, data: DataProto, uid_groups: dict):
        """
        为roleplay数据进行配对处理
        
        Args:
            data: DataProto对象
            uid_groups: uid到索引列表的映射：
            {
            'uuid-aaaa-1111': [0, 2, 4, 6, 8, 10, 12, 14],  # 8个索引
            'uuid-bbbb-2222': [1, 3, 5, 7, 9, 11, 13, 15]   # 8个索引
            }
            
        Returns:
            list: 配对信息列表
        """
        paired_responses = []
        
        for uid, indices in uid_groups.items():
            ### 验证 ###
            # 确保每个uid有n个响应
            expected_responses = self.get_expected_responses()
            if len(indices) != expected_responses:
                print(f"Warning: UID {uid} has {len(indices)} responses, expected {expected_responses}")
                continue
                
            # 从第一个响应获取角色信息
            first_idx = indices[0]
            data_item = data[first_idx]
            data_source = data_item.non_tensor_batch['data_source']
            
            if data_source != 'roleplay':
                continue
            
            ### 提取关键信息 ###
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            role_name = extra_info.get('role_name', '')
            first_category = extra_info.get('first_category', '') 
            prompt_str = extra_info.get('prompt_str', '')
            
            # 解码prompt用于提取角色信息
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            decoded_prompt = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # 如果prompt_str为空，则使用解码后的prompt，prompt_str 用于调用 extract_role_info 函数提取角色信息
            if not prompt_str:
                prompt_str = decoded_prompt
            
            # 提取所有响应文本
            response_texts = []
            for idx in indices:# indices = [0, 2, 4, 6, 8, 10, 12, 14]
                data_item = data[idx]
                
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                
                response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                response_texts.append(response_text)
            
            # 配对：每两个连续的响应组成一对
            # 两两配对：0-1, 2-3, 4-5, ..., 14-15
            for i in range(0, len(response_texts), 2):
                if i + 1 < len(response_texts):
                    pair_info = {
                        'uid': uid,
                        'index_a': indices[i],
                        'index_b': indices[i + 1],
                        'response_a': response_texts[i],
                        'response_b': response_texts[i + 1],
                        'role_name': role_name,
                        'first_category': first_category,
                        'prompt_str': prompt_str
                    }
                    paired_responses.append(pair_info)
                    
        return paired_responses
# 配对结果示例:
# [
#   {'uid': 'uuid-aaaa', 'index_a': 0, 'index_b': 2, ...},   # 配对1
#   {'uid': 'uuid-aaaa', 'index_a': 4, 'index_b': 6, ...},   # 配对2  
#   {'uid': 'uuid-aaaa', 'index_a': 8, 'index_b': 10, ...},  # 配对3
#   {'uid': 'uuid-aaaa', 'index_a': 12, 'index_b': 14, ...}, # 配对4
#   {'uid': 'uuid-bbbb', 'index_a': 1, 'index_b': 3, ...},   # 配对5
#   {'uid': 'uuid-bbbb', 'index_a': 5, 'index_b': 7, ...},   # 配对6
#   {'uid': 'uuid-bbbb', 'index_a': 9, 'index_b': 11, ...},  # 配对7
#   {'uid': 'uuid-bbbb', 'index_a': 13, 'index_b': 15, ...}, # 配对8
# ]

    def compute_paired_rewards(self, data: DataProto, paired_responses: list):
        """
        为配对的响应计算奖励 - 避免重复计算
        新规则：
        1) 两两比较确定胜者；
        2) 对胜者做EOS校验：若无EOS，直接记-1；
        3) 在同一UID组内，对所有通过EOS的胜者按回复长度做线性折减：
           系数 = (组内胜者最短长度) / (该胜者长度)；最短长度对应系数=1
           最终得分 = 胜者基础分(通常为1) * 系数
        非胜者与平局保持compare_responses返回分数（-1或0）。
        """
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # tie 样本掩码（样本级）：True 表示该样本为平局，需要在后续阶段统一屏蔽
        tie_mask = torch.zeros((data.batch['responses'].shape[0],), dtype=torch.bool, device=data.batch['responses'].device)

        total_pairs = len(paired_responses)

        # 统计每个UID的总配对数（仅用于日志）
        uid_pair_counts = defaultdict(int)
        uid_total_pairs = defaultdict(int)
        for pair_info in paired_responses:
            uid_total_pairs[pair_info['uid']] += 1

        # 收集胜者用于事后长度折减：uid -> [ {index, length, base_score} ]
        winners_per_uid = defaultdict(list)
        # 收集立即可落分的项（败者/平局、以及未通过EOS的胜者）
        final_assignments = {}  # idx -> (score, valid_response_length)

        # 便捷函数：获取最后一个有效token位置、长度、末token id
        def _resp_info(item):
            prompt_ids = item.batch['prompts']
            prompt_len = prompt_ids.shape[-1]
            valid_resp_len = item.batch['attention_mask'][prompt_len:].sum()
            response_ids = item.batch['responses'][:valid_resp_len]
            last_token_id = response_ids[-1] if valid_resp_len > 0 else None
            return valid_resp_len, last_token_id

        # 兼容多/单EOS id
        eos_ids = self.tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_set = {eos_ids}
        elif isinstance(eos_ids, (list, tuple, set)):
            eos_set = set(eos_ids)
        else:
            eos_set = set()

        for pair_idx, pair_info in enumerate(paired_responses):
            uid = pair_info['uid']
            index_a = pair_info['index_a']
            index_b = pair_info['index_b']

            # 更新UID配对计数（仅日志）
            uid_pair_counts[uid] += 1
            current_pair_in_uid = uid_pair_counts[uid]
            total_pairs_in_uid = uid_total_pairs[uid]

            data_item_a = data[index_a]
            data_item_b = data[index_b]
            data_source = data_item_a.non_tensor_batch['data_source']

            prompt_preview = pair_info['prompt_str'] 

            self.print(f"\n{'='*100}")
            self.print(f"🎯 Processing New Sample - Pair {pair_idx + 1}/{total_pairs}")
            self.print(f"📋 Data Source: {data_source}")
            self.print(f"🎭 Role: {pair_info['role_name']} | Category: {pair_info['first_category']}")
            self.print(f"🔗 UID Pair: {current_pair_in_uid}/{total_pairs_in_uid} for UID {uid}")
            self.print(f"📍 Response Indices: [{index_a}, {index_b}]")
            self.print(f"💬 Prompt Preview:\n{prompt_preview}")
            self.print(f"{'='*100}")

            self.print(f"\n[Response A - Index {index_a}]")
            self.print(f"{pair_info['response_a']}" )
            self.print(f"\n[Response B - Index {index_b}]")
            self.print(f"{pair_info['response_b']}" )

            from verl.utils.reward_score.roleplay import extract_role_info, compare_responses
            role_info = extract_role_info(pair_info['prompt_str'])

            self.print(f"\n🔍 Comparing responses...")
            
            # TODO:: 可能是平分的来源原因
            # 调用裁判模型，进行打分，需要调转位置后判别仍然相同才会给分，否则平分
            score_a, score_b = compare_responses(
                pair_info['response_a'],
                pair_info['response_b'],
                pair_info['first_category'],
                role_info
            )
            self.print(f"Response A Score(raw): {score_a:.3f}")
            self.print(f"Response B Score(raw): {score_b:.3f}")
            self.print(f"Winner(raw): {'A' if score_a > score_b else 'B' if score_b > score_a else 'Tie'}")
            is_tie = (score_a == score_b)

            # 计算长度与确认EOS正确
            valid_len_a, last_tok_a = _resp_info(data_item_a)
            valid_len_b, last_tok_b = _resp_info(data_item_b)
            eos_ok_a = (last_tok_a is not None and int(last_tok_a) in eos_set)
            eos_ok_b = (last_tok_b is not None and int(last_tok_b) in eos_set)

            # 先处理非胜者/平局：直接落分（-1或0），落在其最后一个有效token
            if score_a < score_b:
                final_assignments[index_a] = (float(score_a), valid_len_a)
            elif is_tie:
                final_assignments[index_a] = (0.0, valid_len_a)
                tie_mask[index_a] = True
            # 若A胜，分数稍后按规则计算

            if score_b < score_a:
                final_assignments[index_b] = (float(score_b), valid_len_b)
            elif is_tie:
                final_assignments[index_b] = (0.0, valid_len_b)
                tie_mask[index_b] = True
            # 若B胜，分数稍后按规则计算

            # 胜者处理：先做EOS校验，未通过则直接-1；通过则收集，稍后按组做长度折减
            if score_a > score_b:
                if not eos_ok_a:
                    self.print("A winner but no EOS -> assign -1")
                    final_assignments[index_a] = (-1.0, valid_len_a)
                else:
                    self.print("A winner has EOS")
                    winners_per_uid[uid].append({'index': index_a, 'length': int(valid_len_a), 'base': float(score_a)})
            if score_b > score_a:
                if not eos_ok_b:
                    self.print("B winner but no EOS -> assign -1")
                    final_assignments[index_b] = (-1.0, valid_len_b)
                else:
                    self.print("B winner has EOS")
                    winners_per_uid[uid].append({'index': index_b, 'length': int(valid_len_b), 'base': float(score_b)})

        # 组内长度线性折减：最短=1，其它=最短/当前长度
        for uid, winners in winners_per_uid.items():
            if not winners:
                continue
            min_len = min(w['length'] for w in winners if w['length'] > 0) if any(w['length'] > 0 for w in winners) else 0
            for w in winners:
                idx = w['index']
                L = max(1, w['length'])
                coef = (min_len / L) if min_len > 0 else 1.0
                final_score = w['base'] * coef
                # 找到该索引的有效长度（用于写入最后token位置）
                # winners里已保存长度，直接使用
                final_assignments[idx] = (final_score, L)
                self.print(f"[UID {uid}] winner idx={idx}, len={L}, min_len={min_len}, coef={coef:.4f}, final_score={final_score:.4f}")

        # 统一写回 reward_tensor（最后一个有效token处）
        for idx, (score, vlen) in final_assignments.items():
            if vlen and vlen > 0:
                # 设置对应idx的最后一个 token 为score
                reward_tensor[idx, vlen - 1] = float(score)

        self.print(f"✅ All {total_pairs} pairs processed with EOS-check and length-based scaling.")
        data.batch['tie_mask'] = tie_mask
        return reward_tensor

    def hard_format_reward(self, model_answer):
        """硬匹配符合thinking"""
        pattern = re.compile(
            r"^\s*"
            r"<think>\s*.+?\s*</think>\s*"
            r"<answer>\s*.+?\s*</answer>\s*"
            r"$",
            re.S
        )

        if pattern.match(model_answer):
            return 1.0
        else:
            return 0.0


    def soft_format_reward(self, model_answer):
        text = model_answer
        score = 0.0

        # 1. 是否包含标签
        has_think = "<think>" in text and "</think>" in text
        has_answer = "<answer>" in text and "</answer>" in text

        if has_think:
            score += 0.2
        if has_answer:
            score += 0.2

        # 2. 标签位置（顺序）
        if has_think and has_answer:
            if text.index("<think>") < text.index("<answer>"):
                score += 0.1

        # 3. 是否闭合
        if bool(re.search(r"<think>.*?</think>", text, re.S)):
            score += 0.2
        if bool(re.search(r"<answer>.*?</answer>", text, re.S)):
            score += 0.2

        # 4. 内容是否为空
        think_content = re.findall(r"<think>(.*?)</think>", text, re.S)
        if think_content and think_content[0].strip():
            score += 0.1

        answer_content = re.findall(r"<answer>(.*?)</answer>", text, re.S)
        if answer_content and answer_content[0].strip():
            score += 0.2


        return score

    def __call__(self, data: DataProto, has_thinking=False, use_soft_format=True):
        """
        Return:
            reward_tensor: 与数据集response(tonken化)相同尺度的奖励张量，原句子最后一个有效token位置设置为得分
            new_samples: []空列表
        """
        # 1.如果存在rm_scores,我们直接返回rm_scores.否则,我们通过rm_score_fn计算
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        ##### validate #####
        if data.meta_info.get('validate', False):
            reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
            import numpy as _np
            N = len(data)
            # 仅统计roleplay；非roleplay样本掩码为False
            mask_b1 = _np.zeros(N, dtype=bool)
            mask_b2 = _np.zeros(N, dtype=bool)
            scores_b1 = _np.full(N, _np.nan, dtype=float)
            scores_b2 = _np.full(N, _np.nan, dtype=float)

            roleplay_total = 0
            found_b1 = found_b2 = 0
            miss_b1 = miss_b2 = 0

            for i in range(len(data)):
                data_item = data[i]
                data_source = data_item.non_tensor_batch['data_source']
                if data_source != 'roleplay':
                    continue

                roleplay_total += 1
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                model_answer = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                extra_info = data_item.non_tensor_batch.get('extra_info', {}) or {}
                role_name = extra_info.get('role_name', '')
                # 从 prompt 里面提取角色名，角色描述和问题
                role_info = extract_role_info(prompt_str)
                if not role_name:
                    role_name = role_info.get('role_name', '')
                role_desc = role_info.get('description', '')
                question = role_info.get('user_question', '')

                # baseline1: ROLEPLAY_BASELINE_JSONL
                b1_ans = self.roleplay_baseline1.get((role_name, question), None) if hasattr(self, 'roleplay_baseline1') else None
                if b1_ans:
                    outcome = self._judge_model_vs_baseline(role_name, role_desc, question, model_answer, b1_ans)
                    # 胜=2，负/平=0；分数落在末token
                    reward_tensor[i, valid_response_length - 1] = 2.0 if outcome == 1 else 0.0
                    scores_b1[i] = 2.0 if outcome == 1 else 0.0
                    mask_b1[i] = True
                    found_b1 += 1
                else:
                    miss_b1 += 1

                # baseline2: BASEMODEL_BASELINE_JSONL
                b2_ans = self.roleplay_baseline2.get((role_name, question), None) if hasattr(self, 'roleplay_baseline2') else None
                if b2_ans:
                    outcome2 = self._judge_model_vs_baseline(role_name, role_desc, question, model_answer, b2_ans)
                    scores_b2[i] = 2.0 if outcome2 == 1 else 0.0
                    mask_b2[i] = True
                    found_b2 += 1
                else:
                    miss_b2 += 1

            self.print(f"[val] roleplay total={roleplay_total} | baseline1 found={found_b1}, missing={miss_b1} | baseline2 found={found_b2}, missing={miss_b2}")

            return reward_tensor, {
                'val_mask_b1': mask_b1,
                'val_mask_b2': mask_b2,
                'val_scores_b1': scores_b1,
                'val_scores_b2': scores_b2,
                'val_total_roleplay': roleplay_total,
                'val_found_b1': found_b1, 'val_missing_b1': miss_b1,
                'val_found_b2': found_b2, 'val_missing_b2': miss_b2
            }

        ##### train #####
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        thinking_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        new_samples = []

        # 检查是否有uid字段，如果有则进行配对处理
        if 'uid' in data.non_tensor_batch:
            # 根据uid分组响应
            uid_groups = self.group_responses_by_uid(data)
            # {
            #   'uuid-aaaa-1111': [0, 2, 4, 6, 8, 10, 12, 14],  # 8个索引
            #   'uuid-bbbb-2222': [1, 3, 5, 7, 9, 11, 13, 15]   # 8个索引
            #    ...
            # } 
            
            # 检查是否有roleplay数据（需要配对处理）
            has_roleplay = False
            for i in range(len(data)):
                data_item = data[i]
                data_source = data_item.non_tensor_batch['data_source'] 
                if data_source == 'roleplay':
                    has_roleplay = True
                    break
            
            if has_roleplay:
                self.print("检测到roleplay数据，开始在naive中进行配对处理...")
                # 仅保留响应数符合预期的uid组（避免配对不齐整）
                valid_uid_groups = {}
                for uid, indices in uid_groups.items():
                    expected_responses = self.get_expected_responses()      # 8
                    if len(indices) == expected_responses:  # 期望每个prompt有n个响应
                        valid_uid_groups[uid] = indices
                    else:
                        print(f"Warning: UID {uid} has {len(indices)} responses, expected {expected_responses}. Skipping pairing for this UID.")
                
                if valid_uid_groups:
                    # 生成两两配对的 response 信息，临近两次生成的组成一对
                    paired_responses = self.pair_responses_for_roleplay(data, valid_uid_groups)
                    self.print(f"配对完成，生成了 {len(paired_responses)} 个配对")
                    
                    # 计算配对奖励（一次比较出两条响应的分数），并把分数落到两条响应的末token
                    reward_tensor = self.compute_paired_rewards(data, paired_responses)
                    
                    # 对于未配对上的响应（例如数量异常），为roleplay直接给0分，保持张量形状一致
                    all_paired_indices = set()
                    for pair_info in paired_responses:
                        all_paired_indices.add(pair_info['index_a'])
                        all_paired_indices.add(pair_info['index_b'])
                    
                    # 处理未配对的响应
                    for i in range(len(data)):
                        if i not in all_paired_indices:
                            data_item = data[i]
                            data_source = data_item.non_tensor_batch['data_source']
                            if data_source == 'roleplay':
                                # 为未配对的roleplay响应使用默认分数
                                prompt_ids = data_item.batch['prompts']
                                prompt_length = prompt_ids.shape[-1]
                                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                                reward_tensor[i, valid_response_length - 1] = 0.0
                    
                    if has_thinking:                    # 添加 thinking 奖励分数
                        for i in range(len(data)):
                            data_item = data[i]
                            prompt_ids = data_item.batch['prompts']
                            prompt_length = prompt_ids.shape[-1]
                            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                            
                            response_ids = data_item.batch['responses']
                            valid_response_ids = response_ids[:valid_response_length]
                            model_answer = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                            
                            if use_soft_format:
                                thinking_tensor[i, valid_response_length - 1] = self.soft_format_reward(model_answer) * 2
                            else:
                                thinking_tensor[i, valid_response_length - 1] = self.hard_format_reward(model_answer) * 2
                    
                    return reward_tensor, new_samples, thinking_tensor

        # 如果没有uid字段或不是roleplay数据，按原有逻辑处理         默认不采用
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode 1. 解码prompt和response
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            
            score, new_sample = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=None,
                prompt_str=prompt_str,
                extra_info=extra_info,
            )
            reward_tensor[i, valid_response_length - 1] = score

            if new_sample:
                new_samples += new_sample

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        return reward_tensor, new_samples

    def get_expected_responses(self):
        if self.config:
            return self.config.actor_rollout_ref.rollout.n
        return 16  # 默认值

    def _create_default_compute_score(self):
        """创建默认的compute_score函数"""
        from verl.trainer.main_ppo import _select_rm_score_fn
        
        def default_compute_score(data_source, solution_str, ground_truth, prompt_str=None, extra_info=None):
            compute_score_fn = _select_rm_score_fn(data_source)
            if data_source == 'roleplay':
                # roleplay需要extra_info参数
                res, new_samples = compute_score_fn(data_source, solution_str, ground_truth, extra_info=extra_info)
            elif 'table' in data_source.lower():
                res, new_samples = compute_score_fn(solution_str, ground_truth, prompt_str, extra_info=extra_info)
            else:
                res, new_samples = compute_score_fn(solution_str, ground_truth, prompt_str)
            
            if isinstance(res, (int, float, bool)):
                return float(res), new_samples
            else:
                return float(res[0]), new_samples
        
        return default_compute_score