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

from verl import DataProto
from verl.workers.reward_manager.tools import is_valid_think_format, remove_think_block, center_distance_linear_scores
import torch
from collections import defaultdict
from verl.utils.embeddings.model import EmbeddingVLLM


class ThinkRewardManager:
    """The reward manager with pairing logic moved from ray_trainer.
    """

    def __init__(self, tokenizer, embedding_url, config=None, p_function=None):
        self.tokenizer = tokenizer
        self.config = config
        self.print = print if p_function is None else p_function
        self.embedding = EmbeddingVLLM(embedding_url)
        
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
        
    def pair_responses(self, data: DataProto, uid_groups: dict):
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
        group_responses = []
        
        for uid, indices in uid_groups.items():
            ### 验证 ###
            # 确保每个uid有n个响应
            expected_responses = self.config.actor_rollout_ref.rollout.n if self.config else 16
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
            extra_info = data_item.non_tensor_batch['extra_info']
            role_name = extra_info['role_name']
            first_category = extra_info['first_category']
            prompt_str = extra_info['prompt_str']
            
            # 提取所有响应文本
            responses = []
            for idx in indices:# indices = [0, 2, 4, 6, 8, 10, 12, 14]
                data_item = data[idx]
                
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                response_ids = data_item.batch['responses'][:valid_response_length]
                last_token_id = response_ids[-1] if valid_response_length > 0 else None
                
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                responses.append({
                    "response": response_text, 
                    "valid_resp_len": valid_response_length, 
                    "last_token_id": last_token_id
                })
            
            group_responses.append({
                "index": indices,
                "responses": responses,
            })
            
            # 配对：每两个连续的响应组成一对
            # 两两配对：0-1, 2-3, 4-5, ..., 14-15
            for i in range(0, len(responses), 2):
                if i + 1 < len(responses):
                    res_a = responses[i]
                    res_b = responses[i + 1]
                    pair_info = {
                        'uid': uid,
                        'index_a': indices[i],
                        'index_b': indices[i + 1],
                        'response_a': res_a["response"],
                        'response_b': res_b["response"],
                        'valid_resp_len_a': res_a["valid_resp_len"],
                        'valid_resp_len_b': res_b["valid_resp_len"],
                        'last_token_id_a': res_a["last_token_id"],
                        'last_token_id_b': res_b["last_token_id"],
                        'role_name': role_name,
                        'first_category': first_category,
                        'prompt_str': prompt_str,
                        'language': extra_info['language']
                    }
                    paired_responses.append(pair_info)
                    
        return paired_responses, group_responses
        
    def pair_responses_for_think(self, data: DataProto, uid_groups: dict):
        """ 仅提取符合格式的样本，并进行配对 """
        paired_responses = []
        group_responses = []
        
        for uid, indices in uid_groups.items():
            # 从第一个响应获取角色信息
            first_idx = indices[0]
            data_item = data[first_idx]
            ### 提取关键信息 ###
            extra_info = data_item.non_tensor_batch['extra_info']
            role_name = extra_info['role_name']
            first_category = extra_info['first_category']
            prompt_str = extra_info['prompt_str']

            # 提取所有响应文本
            responses = []
            new_indices = []
            for idx in indices:         # indices = [0, 2, 4, 6, 8, 10, 12, 14]
                data_item = data[idx]
                
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                response_ids = data_item.batch['responses'][:valid_response_length]
                last_token_id = response_ids[-1] if valid_response_length > 0 else None
                
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                if is_valid_think_format(response_text):
                    responses.append({
                        "response": remove_think_block(response_text), 
                        "valid_resp_len": valid_response_length, 
                        "last_token_id": last_token_id
                    })
                    new_indices.append(idx)
                    
            group_responses.append({
                "index": new_indices,
                "responses": responses ,
            })
            
            for i in range(0, len(responses), 2):
                if i + 1 < len(responses):
                    res_a = responses[i]
                    res_b = responses[i + 1]
                    pair_info = {
                        'uid': uid,
                        'index_a': indices[i],
                        'index_b': indices[i + 1],
                        'response_a': res_a["response"],
                        'response_b': res_b["response"],
                        'valid_resp_len_a': res_a["valid_resp_len"],
                        'valid_resp_len_b': res_b["valid_resp_len"],
                        'last_token_id_a': res_a["last_token_id"],
                        'last_token_id_b': res_b["last_token_id"],
                        'role_name': role_name,
                        'first_category': first_category,
                        'prompt_str': prompt_str,
                        'language': extra_info['language']
                    }
                    paired_responses.append(pair_info)
                    
        return paired_responses, group_responses
        
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

            from verl.utils.reward_score.roleplay import compare_responses
            role_info = data_item_a.non_tensor_batch.get('extra_info')
            role_info['description'] = role_info['role_desc']

            self.print(f"\n🔍 Comparing responses...")
            
            # TODO:: 可能是平分的来源原因
            # 调用裁判模型，进行打分，需要调转位置后判别仍然相同才会给分，否则平分
            score_a, score_b = compare_responses(
                pair_info['response_a'],
                pair_info['response_b'],
                pair_info['first_category'],
                role_info,
                pair_info['language']
            )
            self.print(f"Response A Score(raw): {score_a:.3f}")
            self.print(f"Response B Score(raw): {score_b:.3f}")
            self.print(f"Winner(raw): {'A' if score_a > score_b else 'B' if score_b > score_a else 'Tie'}")
            is_tie = (score_a == score_b)

            # 计算长度与确认EOS正确
            valid_len_a = pair_info['valid_resp_len_a']
            last_tok_a = pair_info['last_token_id_a']
            valid_len_b = pair_info['valid_resp_len_b']
            last_tok_b = pair_info['last_token_id_b']
            
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
        
    def compute_diversity_rewards(self, data: DataProto, group_resonses: list):
        """基于embeddings中心，根据各样本到中心距离进行奖励赋值，最近的为0，最远为1，其余均匀分布"""
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for group in group_resonses:
            indeces = group['index']
            responses = group["responses"]
            
            if len(responses) < self.config.actor_rollout_ref.rollout.n // 2:   # 若样本太少则不进行多样性处理
                self.print("合规样本过少, 跳过 diversity 计算") 
                continue
            
            embeddings = self.embedding.encode([res['response'] for res in responses])
            scores = center_distance_linear_scores(embeddings)
            
            if len(indeces) != len(scores):
                self.print("索引个数与得分数不一致, 跳过 diversity 计算")
                continue
            
            for index, score, response in zip(indeces, scores, responses):
                valid_resp_len = response['valid_resp_len']
                if valid_resp_len and valid_resp_len > 0:
                    reward_tensor[index, valid_resp_len - 1] = float(score)
        
        return reward_tensor
    
    def compute_format_rewards(self, data: DataProto, group_resonses: list):
        """ 格式完全正确赋 1.0 分, 否则 0.0 分 """
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for group in group_resonses:
            indeces = group['index']
            responses = group["responses"]
            
            for index, response in zip(indeces, responses):
                valid_resp_len = response['valid_resp_len']
                if valid_resp_len and valid_resp_len > 0:
                    reward_tensor[index, valid_resp_len - 1] = float(1.0)
        
        return reward_tensor
        
    def __call__(self, data: DataProto, has_thinking=False):
        """
        Return:
            reward_tensor: 与数据集response(tonken化)相同尺度的奖励张量，原句子最后一个有效token位置设置为得分
            new_samples: []空列表
        """
        uid_groups = self.group_responses_by_uid(data)
        
        if has_thinking:        # 过滤掉不符合thinking格式的样本
            paired_responses, group_responses = self.pair_responses_for_think(data, uid_groups)
            thinking_tensor = self.compute_format_rewards(data, group_responses)
        else:
            paired_responses, group_responses = self.pair_responses(data, uid_groups)
            thinking_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        self.print(f"配对完成，生成了 {len(paired_responses)} 个配对")
        
        # 计算配对奖励（一次比较出两条响应的分数），并把分数落到两条响应的末token
        reward_tensor = self.compute_paired_rewards(data, paired_responses)
        diversity_tensor = self.compute_diversity_rewards(data, group_responses)
                
        return reward_tensor, thinking_tensor, diversity_tensor
