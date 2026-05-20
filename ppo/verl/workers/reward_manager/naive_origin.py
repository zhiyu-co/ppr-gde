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
# from verl.utils.reward_score import _default_compute_score
import torch
import numpy as np
from collections import defaultdict
from verl.utils.reward_score.roleplay import compute_score


class NaiveRewardManager:
    """The reward manager with pairing logic moved from ray_trainer.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, config=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # 如果没有传入compute_score，创建一个默认的
        self.compute_score = compute_score or self._create_default_compute_score()
        self.config = config
        self.expected_responses = self.get_expected_responses()  # 新增参数

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

            score, rethink_sample = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                prompt_str=prompt_str,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def group_responses_by_uid(self, data: DataProto):
        """
        根据uid字段对responses进行分组
        
        Args:
            data: DataProto对象
            
        Returns:
            dict: uid -> [response_indices] 的映射
            {'uuid-1': [0,5,12,18,...], 'uuid-2': [3,8,15,22,...]}
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
        
        Args:
            data: DataProto对象
            paired_responses: 配对信息列表
            
        Returns:
            torch.Tensor: 奖励张量
        """
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        total_pairs = len(paired_responses)
        
        # 按UID分组配对，方便显示是第几个prompt的第几对
        uid_pair_counts = defaultdict(int)
        uid_total_pairs = defaultdict(int)
        
        # 先统计每个UID的总配对数
        for pair_info in paired_responses:
            uid_total_pairs[pair_info['uid']] += 1
        
        for pair_idx, pair_info in enumerate(paired_responses):
            uid = pair_info['uid']
            index_a = pair_info['index_a']
            index_b = pair_info['index_b']
            
            # 更新当前UID的配对计数
            uid_pair_counts[uid] += 1
            current_pair_in_uid = uid_pair_counts[uid]
            total_pairs_in_uid = uid_total_pairs[uid]
            
            # 获取数据项信息
            data_item_a = data[index_a]
            data_item_b = data[index_b]
            data_source = data_item_a.non_tensor_batch['data_source']
            
            # 显示详细日志信息
            prompt_preview = pair_info['prompt_str']
            
            print(f"\n{'='*100}")
            print(f"🎯 Processing New Sample - Pair {pair_idx + 1}/{total_pairs}")
            print(f"📋 Data Source: {data_source}")
            print(f"🎭 Role: {pair_info['role_name']} | Category: {pair_info['first_category']}")
            print(f"🔗 UID Pair: {current_pair_in_uid}/{total_pairs_in_uid} for UID {uid[:8]}...")
            print(f"📍 Response Indices: [{index_a}, {index_b}]")
            print(f"💬 Prompt Preview:\n{prompt_preview}")
            print(f"{'='*100}")
            
            # 显示两个回复的内容
            print(f"\n[Response A - Index {index_a}]")
            print(f"{pair_info['response_a']}" if len(pair_info['response_a']) > 200 else pair_info['response_a'])
            print(f"\n[Response B - Index {index_b}]") 
            print(f"{pair_info['response_b']}" if len(pair_info['response_b']) > 200 else pair_info['response_b'])
            
            # 一次性计算两个回复的分数 
            from verl.utils.reward_score.roleplay import extract_role_info, compare_responses
            
            # 提取角色信息
            role_info = extract_role_info(pair_info['prompt_str'])
            
            print(f"\n🔍 Comparing responses...")
            
            # 一次比较获得两个分数 - 避免重复计算！
            score_a, score_b = compare_responses(
                pair_info['response_a'],
                pair_info['response_b'], 
                pair_info['first_category'],
                role_info
            )
            
            print(f"Response A Score: {score_a:.3f}")
            print(f"Response B Score: {score_b:.3f}")
            print(f"Winner: {'Response A' if score_a > score_b else 'Response B' if score_b > score_a else 'Tie'}")
            
            # 分别设置两个回复的奖励
            # 设置回复A的奖励
            prompt_ids_a = data_item_a.batch['prompts']
            prompt_length_a = prompt_ids_a.shape[-1]
            valid_response_length_a = data_item_a.batch['attention_mask'][prompt_length_a:].sum()
            reward_tensor[index_a, valid_response_length_a - 1] = score_a
            
            # 设置回复B的奖励
            prompt_ids_b = data_item_b.batch['prompts']
            prompt_length_b = prompt_ids_b.shape[-1]
            valid_response_length_b = data_item_b.batch['attention_mask'][prompt_length_b:].sum()
            reward_tensor[index_b, valid_response_length_b - 1] = score_b
            
            print(f"✅ Pair {pair_idx + 1}/{total_pairs} completed successfully!")
            print(f"{'='*100}\n")
                
        return reward_tensor

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        # 1.如果存在rm_scores,我们直接返回rm_scores.否则,我们通过rm_score_fn计算
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        new_samples = []

        # 检查是否有uid字段，如果有则进行配对处理
        if 'uid' in data.non_tensor_batch:
            # 根据uid分组响应
            uid_groups = self.group_responses_by_uid(data)
            # {
            #   'uuid-aaaa-1111': [0, 2, 4, 6, 8, 10, 12, 14],  # 8个索引
            #   'uuid-bbbb-2222': [1, 3, 5, 7, 9, 11, 13, 15]   # 8个索引
            # }
            # 检查是否是roleplay数据需要配对处理
            has_roleplay = False
            for i in range(len(data)):
                data_item = data[i]
                data_source = data_item.non_tensor_batch['data_source']
                if data_source == 'roleplay':
                    has_roleplay = True
                    break
            
            if has_roleplay:
                print("检测到roleplay数据，开始在naive中进行配对处理...")
                # 检查每个uid组的大小，确保配对逻辑正确
                valid_uid_groups = {}
                for uid, indices in uid_groups.items():
                    expected_responses = self.get_expected_responses()
                    if len(indices) == expected_responses:  # 期望每个prompt有n个响应
                        valid_uid_groups[uid] = indices
                    else:
                        print(f"Warning: UID {uid} has {len(indices)} responses, expected {expected_responses}. Skipping pairing for this UID.")
                
                if valid_uid_groups:
                    # 为roleplay数据进行配对
                    paired_responses = self.pair_responses_for_roleplay(data, valid_uid_groups)
                    print(f"配对完成，生成了 {len(paired_responses)} 个配对")
                    
                    # 计算配对奖励
                    reward_tensor = self.compute_paired_rewards(data, paired_responses)
                    
                    # 对于没有有效配对的响应，使用默认逻辑处理
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
                    
                    return reward_tensor, new_samples

        # 如果没有uid字段或不是roleplay数据，按原有逻辑处理
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