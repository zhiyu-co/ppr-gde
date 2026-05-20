# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import math
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# # NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
# def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
#                                    eos_mask: torch.Tensor,
#                                    index: torch.Tensor,
#                                    epsilon: float = 1e-6):
#     """
#     response length sensitive
#     Compute advantage for GRPO, operating only on Outcome reward
#     (with only one scalar reward for each response).
#     Args:
#         token_level_rewards: `(torch.Tensor)`
#             shape: (bs, response_length)
#         eos_mask: `(torch.Tensor)`
#             shape: (bs, response_length)
#
#     Returns:
#         advantages: `(torch.Tensor)`
#             shape: (bs, response_length)
#         Returns: `(torch.Tensor)`
#             shape: (bs, response_length)
#     """
#     response_length = token_level_rewards.shape[-1]
#     scores = token_level_rewards.sum(dim=-1)
#
#     id2score = defaultdict(list)
#     id2mean = {}
#     id2std = {}
#     id2mask = defaultdict(list)
#     id2maskmin = {}
#     id2maskmax = {}
#
#     with torch.no_grad():
#         bsz = scores.shape[0]
#         for i in range(bsz):
#             id2score[index[i]].append(scores[i])
#             id2mask[index[i]].append(eos_mask[i].sum())
#         for idx in id2mask:
#             id2maskmin[idx] = min(id2mask[idx])
#             id2maskmax[idx] = max(id2mask[idx])
#         for idx in id2score:
#             if len(id2score[idx]) > 1:
#                 for jdx in range(len(id2score[idx])):
#                     res_score = id2score[idx][jdx]
#                     res_len = id2mask[idx][jdx]
#                     # score cosine scaling decay
#                     progress = res_len / id2maskmax[idx]
#                     cosine = math.cos(progress*math.pi)
#                     res_score += 0.1*(1.0+cosine)
#                     id2score[idx][jdx] = res_score
#
#         for idx in id2score:
#             if len(id2score[idx]) == 1:
#                 id2mean[idx] = torch.tensor(0.0)
#                 id2std[idx] = torch.tensor(1.0)
#             elif len(id2score[idx]) > 1:
#                 id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
#                 id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
#             else:
#                 raise ValueError(f"no score in prompt index: {idx}")
#         for i in range(bsz):
#             scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
#             # scores[i] = (scores[i] - id2mean[index[i]])  # no std
#         scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
#
#     return scores, scores


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
# def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
#                                    eos_mask: torch.Tensor,
#                                    index: torch.Tensor,
#                                    epsilon: float = 1e-6):
#     """
#     name: lgpo long-term group
#     assert mean-score = 0.0
#     Compute advantage for GRPO, operating only on Outcome reward
#     (with only one scalar reward for each response).
#     Args:
#         token_level_rewards: `(torch.Tensor)`
#             shape: (bs, response_length)
#         eos_mask: `(torch.Tensor)`
#             shape: (bs, response_length)
#
#     Returns:
#         advantages: `(torch.Tensor)`
#             shape: (bs, response_length)
#         Returns: `(torch.Tensor)`
#             shape: (bs, response_length)
#     """
#     response_length = token_level_rewards.shape[-1]
#     scores = token_level_rewards.sum(dim=-1)
#
#     id2score = defaultdict(list)
#     id2absscore = defaultdict(list)
#     id2std = {}
#     id2mask = defaultdict(list)
#     id2sum = {}
#
#     with torch.no_grad():
#         bsz = scores.shape[0]
#         for i in range(bsz):
#             id2score[index[i]].append(scores[i])
#             id2mask[index[i]].append(eos_mask[i].sum())
#             id2absscore[index[i]].append(abs(scores[i]))
#
#         for idx in id2score:
#             if len(id2score[idx]) == 1:
#                 id2std[idx] = torch.tensor(1.0)
#                 id2sum[idx] = torch.tensor(id2absscore[idx])
#             elif len(id2score[idx]) > 1:
#                 id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
#                 id2sum[idx] = torch.sum(torch.tensor([id2absscore[idx]]))
#             else:
#                 raise ValueError(f"no score in prompt index: {idx}")
#         for i in range(bsz):
#             # scores[i] = (scores[i]) / (id2std[index[i]] + epsilon)
#             # scores[i] = scores[i]  # without std
#             scores[i] = (scores[i]) / (id2sum[index[i]] + epsilon)
#
#         scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
#
#     return scores, scores


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]              # 每条样本的回复长度（token数）
    scores = token_level_rewards.sum(dim=-1)                     # 将token级奖励按序列求和，得到每条样本的序列得分（标量）

    id2score = defaultdict(list)                                 # uid -> 该uid下的所有序列得分列表
    id2mean = {}                                                 # uid -> 该uid组内的均值
    id2std = {}                                                  # uid -> 该uid组内的标准差
    id2mask = defaultdict(list)                                  # uid -> 该uid下每条样本的有效回复长度（当前版本未参与归一化，仅记录）

    with torch.no_grad():                                        # 仅统计与归一化操作，不需要梯度
        bsz = scores.shape[0]                                    # batch 大小
        for i in range(bsz):
            id2score[index[i]].append(scores[i])                 # 依据 index[i]（即uid）将序列得分归入对应组
            id2mask[index[i]].append(eos_mask[i].sum())          # 记录该样本的有效回复长度，备用

        for idx in id2score:                                     # 逐个uid组计算组内统计量
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)                 # 单样本组：均值置0
                id2std[idx] = torch.tensor(1.0)                  # 单样本组：标准差置1，避免除0
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))   # 多样本组：组内均值
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))   # 多样本组：组内标准差
            else:
                raise ValueError(f"no score in prompt index: {idx}")     # 理论上不会发生

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)  # 组内标准化（去均值、除方差）
            # scores[i] = (scores[i] - id2mean[index[i]])  # no std                     # 仅去均值（备选实现）

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask            # 将序列标量广播回token维，并按mask裁剪

    return scores, scores                                         # 在该实现中，优势=回报（仅基于outcome标量）


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                        1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    eos_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):  # 定义聚合损失为标量的函数
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":  # 模式1：对所有 token 的损失做全局（被 mask 的位置忽略）平均
        loss = verl_F.masked_mean(loss_mat, loss_mask)  # 按 mask 求均值：等价于 sum(loss*mask)/sum(mask)
    elif loss_agg_mode == "seq-mean-token-sum":  # 模式2：先对每条序列的 token 损失求和，再对序列做平均
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # 按最后一维(token维)求和，得到每条序列的损失（token-sum）
        loss = torch.mean(seq_losses)  # 对 batch 内所有序列的和再做平均（seq-mean）
    elif loss_agg_mode == "seq-mean-token-mean":  # 模式3：先对每条序列的 token 损失求均值，再对序列做平均
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # 每条序列：sum(loss)/有效token数
        loss = torch.mean(seq_losses)  # 对 batch 内序列的 token-mean 再做平均（seq-mean）
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")  # 非法模式报错

    return loss  # 返回标量损失


# def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
#     """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
#
#     Args:
#         old_log_prob: `(torch.Tensor)`
#             shape: (bs, response_length)
#         log_prob: `(torch.Tensor)`
#             shape: (bs, response_length)
#         advantages: `(torch.Tensor)`
#             shape: (bs, response_length)
#         eos_mask: `(torch.Tensor)`
#             shape: (bs, response_length)
#         cliprange: (float)
#             The clip range used in PPO. See https://arxiv.org/abs/1707.06347
#
#     Returns:
#         pg_loss: `a scalar torch.Tensor`
#             policy gradient loss computed via PPO
#         pg_clipfrac: (float)
#             a float number indicating the fraction of policy gradient loss being clipped
#
#     """
#     negative_approx_kl = log_prob - old_log_prob
#     ratio = torch.exp(negative_approx_kl)
#     ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)
#
#     pg_losses = -advantages * ratio
#     pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
#
#     # token-mean
#     # grpo loss
#     grpo_pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
#
#     # ggrpo loss
#     pg_losses3 = torch.max(pg_losses, pg_losses2)
#     pg_losses3_better = torch.masked_select(pg_losses3, advantages > 0)
#     eos_mask_better = torch.masked_select(eos_mask, advantages > 0)
#
#     pg_losses3_worse = torch.masked_select(pg_losses3, advantages < 0)
#     eos_mask_worse = torch.masked_select(eos_mask, advantages < 0)
#
#     pg_losses3_better = verl_F.masked_mean(pg_losses3_better, eos_mask_better)
#     pg_losses3_worse = verl_F.masked_mean(pg_losses3_worse, eos_mask_worse)
#     pg_loss = (pg_losses3_better + pg_losses3_worse)/2
#
#     pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
#
#     print("ggrpo loss, grpo pg_loss, pg_clipfrac, ppo_kl:", pg_loss, grpo_pg_loss, pg_clipfrac, ppo_kl)
#
#     return pg_loss, pg_clipfrac, ppo_kl


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):  # 计算 PPO 策略损失
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob  # 逐 token 的对数概率差：log π - log π_old（近似 -KL 的负号项）
    ratio = torch.exp(negative_approx_kl)  # 概率比 r = exp(logπ - logπ_old) = π/π_old（逐元素）
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)  # 近似 KL: E_old[log π_old - log π]，按 mask 平均

    pg_losses = -advantages * ratio  # 未裁剪的 PG 损失：-A * r（注意 A>0 时希望 r 大；损失越小越好）
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)  # 裁剪版：-A * clip(r)

    # # token-mean
    # pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)

    # # seq-mean-token-mean
    # # seq_losses = torch.sum(torch.max(pg_losses, pg_losses2) * eos_mask, dim=-1) / torch.sum(eos_mask, dim=-1)   
    
    # token-mean
    # # pg_loss = torch.mean(seq_losses)  # seq-mean
    # seq-mean-token-mean：先每条序列对 token 均值，再对序列均值（与全局 token-mean 略有差异）
    pg_losses_agg_src = torch.max(pg_losses, pg_losses2)  # PPO-clip：取两者较大（更保守的损失）
    pg_loss = agg_loss(loss_mat=pg_losses_agg_src, loss_mask=eos_mask, loss_agg_mode="seq-mean-token-mean")  # 聚合为标量

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)  # 被裁剪比例：clip 版优于未裁剪的占比

    # TODO:::::: delete
    print(f"grpo pg_loss: {pg_loss.item()} \tpg_clipfrac: {pg_clipfrac.item()} \tppo_kl: {ppo_kl.item()}")  # 打印监控指标

    # print("grpo pg_loss, pg_clipfrac, ppo_kl:", pg_loss, pg_clipfrac, ppo_kl)  # 重复打印（原始代码保留）

    return pg_loss, pg_clipfrac, ppo_kl  # 返回：策略损失、裁剪比例、近似 KL


def dapo_compute_policy_loss(old_log_prob,                         # 双重裁剪 PPO（Dual-clip PPO）版本的策略损失
                        log_prob,
                        advantages,
                        response_mask,
                        cliprange=None,
                        cliprange_low=None,
                        cliprange_high=None,
                        clip_ratio_c=3.0,
                        loss_agg_mode="token-mean"):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."  # dual-clip 的额外比值下界常数 c>1

    negative_approx_kl = log_prob - old_log_prob  # 逐 token 的对数概率差
    ratio = torch.exp(negative_approx_kl)  # 概率比 r = π/π_old
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)  # 近似 KL

    pg_losses1 = -advantages * ratio  # 未裁剪的 PG 损失
    if cliprange_low is None:  # 支持非对称裁剪：下界
        cliprange_low = cliprange
    if cliprange_high is None:  # 支持非对称裁剪：上界
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low,
                                           1 + cliprange_high)  # 裁剪版 PG 损失（按上下界分别约束）
    clip_pg_losses1 = torch.maximum(pg_losses1,
                                    pg_losses2)  # 常规 PPO：取两者较大（更保守）。maximum 是逐元素较大值
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)  # 常规裁剪比例

    pg_losses3 = -advantages * clip_ratio_c  # dual-clip 的第二重裁剪：对负优势样本引入下界 c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)  # 逐元素最小值：当 A<0 时强行限制最大损失规模
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses2, pg_losses3) * (advantages < 0).float(), response_mask)  # 仅在 A<0 的样本上统计第二重裁剪比例

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)  # A<0 用 dual-clip，A>=0 用常规 clip
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)  # 聚合为标量（支持三种模式）

    print(f"grpo pg_loss: {pg_loss.item()} \tpg_clipfrac: {pg_clipfrac.item()} \tppo_kl: {ppo_kl.item()}")  # 打印监控指标

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower  # 额外返回负优势样本的裁剪比例


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
