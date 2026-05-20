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
from collections import defaultdict
import numpy as np

from verl import DataProto
from verl.utils.embeddings.model import EmbeddingVLLM
from verl.workers.reward_manager.tools import (
    is_valid_think_format, 
    remove_think_block, 
    center_distance_linear_scores, 
    get_message,
    predict
)


class GRPORewardManager:
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
        
    def group_responses(self, data: DataProto, uid_groups: dict):
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
                "uid": uid,
                "prompt_str": prompt_str,
                "role_name": role_name,
                "first_category": first_category,
                "language": extra_info['language'],
                "index": indices,
                "responses": responses
            })
                    
        return group_responses
        
    def group_responses_for_think(self, data: DataProto, uid_groups: dict):
        """ 仅提取符合格式的样本，并进行配对 """
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
                "uid": uid,
                "prompt_str": prompt_str,
                "role_name": role_name,
                "first_category": first_category,
                "language": extra_info['language'],
                "index": new_indices,
                "responses": responses
            })
                    
        return group_responses
        
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
    
    def compute_grpo_rewards(self, data, group_responses):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for res in group_responses:
            language = res['language']
            scores = []
            
            for index, response in zip(res['index'], res['responses']):
                data_item = data[index]
                role_info = data_item.non_tensor_batch['extra_info']
                role_info['description'] = role_info['role_desc']
                role_name = role_info['role_name']
                role_desc = role_info['description']
                user_question = role_info['user_question']
                
                answer = response['response']
                valid_resp_len = response['valid_resp_len']
                
                message = get_message(role_name, role_desc, user_question, answer, language)
                
                score = predict(message)
                
                scores.append(score)
            
            if len(scores) != len(res['index']):
                print("分数生成错误, 跳过当前组")
                continue
            
            scores = np.array(scores)
            
            for index, response, score in zip(res['index'], res['responses'], scores):
                valid_resp_len = response['valid_resp_len']
                if valid_resp_len and valid_resp_len > 0:
                    reward_tensor[index, valid_resp_len - 1] = float(score)
        
        return reward_tensor
    
    def __call__(self, data: DataProto, has_thinking=False):
        """
        Return:
            reward_tensor: 与数据集response(tonken化)相同尺度的奖励张量，原句子最后一个有效token位置设置为得分
            new_samples: []空列表
        """
        uid_groups = self.group_responses_by_uid(data)
        
        if has_thinking:        # 过滤掉不符合thinking格式的样本
            group_responses = self.group_responses_for_think(data, uid_groups)
            thinking_tensor = self.compute_format_rewards(data, group_responses)
        else:
            group_responses = self.group_responses(data, uid_groups)
            thinking_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        
        # 计算配对奖励（一次比较出两条响应的分数），并把分数落到两条响应的末token
        reward_tensor = self.compute_grpo_rewards(data, group_responses)
        diversity_tensor = self.compute_diversity_rewards(data, group_responses)
                
        return reward_tensor, thinking_tensor, diversity_tensor