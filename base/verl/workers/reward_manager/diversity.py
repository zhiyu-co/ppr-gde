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
import json
import numpy as np
import re
import os
from collections import defaultdict
from typing import List
from sklearn.metrics.pairwise import cosine_distances
from verl.utils.diversity import (
    compression_ratio,
    rougel_score,
    distinct_n_scores
)

from verl.utils.uncertainty.semantic_entropy import (cluster_assignment_entropy, 
                                                        predictive_entropy, 
                                                        predictive_entropy_rao, 
                                                        get_semantic_ids_by_embedding, 
                                                        logsumexp_by_id)
from verl.utils.embeddings.model import EmbeddingVLLM
from verl.workers.reward_manager.tools import remove_think_block


def compute_log_liks_agg_group(responses, log_liks, tokenizer):
    """
    responses: List[str]
    log_liks:  List[List[float]]
    tokenizer: HF tokenizer, with .encode() or .tokenize()
    """
    cleaned_texts = [remove_think_block(r) for r in responses]

    cleaned_log_liks = []
    for text, ll in zip(cleaned_texts, log_liks):
        # 重新 tokenize 清洗后的 answer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        L = len(tokens)

        if L == 0:
            # 若 answer 内容为空，fallback 使用原 log_lik 的平均
            cleaned_log_liks.append(np.mean(ll))
        else:
            # log_lik 末尾的 L 个 token 对应 answer 部分
            cleaned_ll = ll[-L:]
            cleaned_log_liks.append(np.mean(cleaned_ll))

    return cleaned_log_liks, cleaned_texts


def compute_log_liks_agg(response, log_lik, tokenizer):
    """
    responses: List[str]
    log_liks:  List[List[float]]
    tokenizer: HF tokenizer, with .encode() or .tokenize()
    """
    cleaned_text = remove_think_block(response)

    tokens = tokenizer.encode(cleaned_text, add_special_tokens=False)
    L = len(tokens)
    if L == 0:
        # 若 answer 内容为空，fallback 使用原 log_lik 的平均
        return log_lik, response
    else:
        return log_lik[-L:], cleaned_text


class DiversityRewardManager:
    """The reward manager with pairing logic moved from ray_trainer.
    """

    def __init__(self, tokenizer, embedding_url, config=None):
        self.tokenizer = tokenizer
        self.config = config
        self.embedding = EmbeddingVLLM(embedding_url)

    def embedding_diversity(self, data: List[str], verbose=False, batch_size = 64):
        """
        remote_clique: 每个样本和其余样本的距离均值
        chamfer_dist: 每个样本和其余样本中最相似样本的距离均值
        """
        # [N, 768] embedding
        embeddings = self.embedding.encode(data, batch_size=batch_size, show_progress_bar=verbose)
        distances = cosine_distances(embeddings)
        
        mean_distances = np.mean(distances, axis=1)
        remote_clique = np.mean(mean_distances).round(3)
        
        min_distances = np.min(distances + np.eye(len(distances)) * 1e9, axis=1)
        chamfer_dist = np.mean(min_distances).round(3)
        
        return remote_clique, chamfer_dist
    
    def diversity(self, texts):
        score = {}
        # Compression ratio
        score["compression_ratio"] = compression_ratio(texts, algorithm='gzip')

        # Homogenization score (Self-BLEU)
        score["rouge-L"] = rougel_score(texts, use_stemmer=True)
        
        # N-gram diversity
        res = distinct_n_scores(texts, self.tokenizer)
        score["distinct-1"] = res["distinct-1"]
        score["distinct-2"] = res["distinct-2"]
        
        remote_clique, chamfer_dist = self.embedding_diversity(texts)
        score["remote_clique"] = remote_clique
        score["chamfer_dist"] = chamfer_dist
        return score

    def __call__(self, data: DataProto, step=0, has_thinking=False):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        log_dict = {}
        
        for i in range(len(data)):
            data_item = data[i]
            
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            valid_response_ids = data_item.batch['responses'][:valid_response_length]

            model_answer = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            extra_info = data_item.non_tensor_batch['extra_info']
            question = extra_info['user_question']
            uid = data_item.non_tensor_batch['uid']
            
            log_likelihoods = data.batch['log_likelihoods'][:valid_response_length]
            
            if has_thinking:
                log_likelihoods, only_answer = compute_log_liks_agg(model_answer, log_likelihoods, self.tokenizer)
            
            if uid not in log_dict:
                log_dict[uid] = {
                    "uid": uid,
                    "question": question,
                    "language": extra_info["language"],
                    "answer": [],
                    "log_liks": []
                }
                if has_thinking:
                    log_dict[uid].update({"ori_answer": []})
            
            log_dict[uid]["log_liks"].append(np.mean(np.array(log_likelihoods)))
            
            if has_thinking:
                log_dict[uid]["answer"].append(only_answer)
                log_dict[uid]["ori_answer"].append(model_answer)
            else:
                log_dict[uid]["answer"].append(model_answer)
            
        diversity_score = {}
        
        print("✅ 开始进行多样性评估!")
        log_print = []
        
        for uid, data in log_dict.items():
            l = len(data["answer"])
            if l != self.config.actor_rollout_ref.rollout.val_kwargs.n:
                print("=" * 20 + f"生成数量不足，跳过uid: {uid}  长度 {l}!")
                continue
            log_liks_agg = data["log_liks"]
            responses = data["answer"]
            question = data["question"]
            
            res = self.diversity(responses)
            diversity_score.setdefault("compression_ratio", []).append(res["compression_ratio"])
            diversity_score.setdefault("distinct-1", []).append(res["distinct-1"])
            diversity_score.setdefault("distinct-2", []).append(res["distinct-2"])
            diversity_score.setdefault("rouge-L", []).append(res["rouge-L"])
            diversity_score.setdefault("remote_clique", []).append(res["remote_clique"])
            diversity_score.setdefault("chamfer_dist", []).append(res["chamfer_dist"])
            
            # 计算每个response的熵
            regular_entropy = predictive_entropy(log_liks_agg)
            diversity_score.setdefault("regular_entropy", []).append(regular_entropy)
            
            # 语义聚类
            semantic_ids = get_semantic_ids_by_embedding(responses, model=self.embedding)
            
            num_clusters = len(set(semantic_ids))
            diversity_score.setdefault("num_clusters", []).append(num_clusters)
            
            # 簇分配熵，即将每个聚类作为一个输出可能，计算模型在语意层面的多样性，不考虑概率
            # -sum(n_c/N * log(n_c/N))  n_c=簇内条数    N=总条数
            cluster_assignment_entropys = cluster_assignment_entropy(semantic_ids)
            diversity_score.setdefault("cluster_assignment_entropys", []).append(cluster_assignment_entropys)
            
            # 聚合同簇聚类的 总概率
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
            # 语义熵 = -sum(P(c) * log(P(c)))   P(c)=簇的生成概率
            # 不仅考虑多样性，还考虑模型置信度，即如果模型对聚类概率越平均，熵越大（不确定性越高）
            semantic_entropy = predictive_entropy_rao(log_likelihood_per_semantic_id)
            diversity_score.setdefault("semantic_entropy", []).append(semantic_entropy)
            
            tem = {
                "uid": uid, 
                "question": question, 
                "num_clusters": float(num_clusters),
                "semantic_entropy": float(semantic_entropy),
                "cluster_assignment_entropys": float(cluster_assignment_entropys),
                "regular_entropy": float(regular_entropy),
                "compression_ratio": res["compression_ratio"],
                "distinct-1": res["distinct-1"],
                "distinct-2": res["distinct-2"],
                "rouge-L": res["rouge-L"],
                "remote_clique": res["remote_clique"],
                "chamfer_dist": res["chamfer_dist"],
                "answer": responses
            }
            if has_thinking:
                tem["ori_answer"] = data["ori_answer"]
            
            log_print.append(tem)
        
        
        for k in list(diversity_score.keys()):
            diversity_score[k] = np.array(diversity_score[k], dtype=float)

        log_dir = os.getenv("LOG_DIR", ".")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"diversity_results_{step}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(list(log_print), f, ensure_ascii=False, indent=2)
        
        return diversity_score

