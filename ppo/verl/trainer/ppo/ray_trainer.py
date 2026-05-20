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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import threading
import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from time import time

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader


from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.logger.logger import Logger


WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    """
    return:
        data: 更新过分数的数据
        metrics: 记录 kl 散度和 beta 系数的矩阵
    """
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]    # 从句末提取 response 的 mask

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value        # KL控制器的值，即散度系数
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    # 先计算每个样本的kl均值，然后计算整个batch的均值
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    # 更新kl控制器, 一个样本对应走了一步，所以 n_steps = batch_size
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards     # 更新记录

    # metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}
    metrics = {'actor/reward_kl_penalty': current_kl, 'actor/reward_kl_penalty_coeff': beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    """
    按照 uid 应用 GAE 对 token-level reward 进行分组计算
    data.batch['advantages']: 用来更新策略（优化模型）
    data.batch['returns']: 用来更新价值函数（critic）
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch['response_mask'] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if 'tie_mask' in data.batch:
        tie_mask = data.batch['tie_mask']
        if isinstance(tie_mask, torch.Tensor):
            rm = data.batch['response_mask']
            data.batch['response_mask'] = rm * (~tie_mask).to(dtype=rm.dtype).unsqueeze(1)
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        # responses = data.batch['responses']
        # response_length = responses.size(-1)
        # attention_mask = data.batch['attention_mask']
        # response_mask = attention_mask[:, -response_length:]
        response_mask = data.batch['response_mask']
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def compute_reward_metrics(batch):
    """ 
    返回 reward_metrics 用于给 wandb 进行记录
    ["reward/mean"]     ["reward/win_answer_ratio"]     ["reward/lose_answer_ratio"]    ["reward/tie_ratio"]
    """
    reward_tensor = batch.batch['reward_scores'].sum(-1)   # 每条样本的总 reward
    reward_metrics = {}
    reward_metrics["reward/quality_mean"] = torch.mean(reward_tensor).detach().item()   # 所有样本的奖励均值
    
    # 适配roleplay的分数范围 {-1, 0, 1}
    win_answer_ratio = torch.sum(reward_tensor > 0).float() / reward_tensor.numel()
    reward_metrics["reward/win_answer_ratio"] = win_answer_ratio.detach().item()    # reward > 0 个数
    
    lose_answer_ratio = torch.sum(reward_tensor < 0).float() / reward_tensor.numel()
    reward_metrics["reward/lose_answer_ratio"] = lose_answer_ratio.detach().item()  # reward < 0 个数
    
    # 平局比例
    tie_ratio = torch.sum(reward_tensor == 0).float() / reward_tensor.numel() 
    reward_metrics["reward/tie_ratio"] = tie_ratio.detach().item()                  # reward = 0 个数
    
    quality_tensor = batch.batch['token_level_scores'].sum(-1)
    reward_metrics["reward/reward_mean"] = torch.mean(quality_tensor).detach().item()
    
    think_tensor = batch.batch['thinking_scores'].sum(-1)
    reward_metrics["reward_score/thinking_mean"] = torch.mean(think_tensor).detach().item()
    
    diversity_tensor = batch.batch['diversity_scores'].sum(-1)
    reward_metrics["reward/diversity_mean"] = torch.mean(diversity_tensor).detach().item()
    
    return reward_metrics


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 p_function=None,
                 role_llm=None,
                 char_rm=None,
                 diversity=None,
                 has_thinking=False,
                 ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'
        self.print = print if p_function is None else p_function
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.role_llm = role_llm
        self.char_rm = char_rm
        self.diversity = diversity
        self.thinking = has_thinking

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _update_train_dataloader(self, all_rethink_samples):
        self.train_dataset.add_new_data(all_rethink_samples)

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        assert len(self.train_dataloader) >= 1

        print(f'Size of new train dataloader: {len(self.train_dataloader)}')

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error',
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)
        
        # TODO:::::::::::::::
        self.char_dataset = RLHFDataset(parquet_files=self.config.data.val_char_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_char_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error',
                                       filter_overlong_prompts=False)
        self.char_dataloader = StatefulDataLoader(
            dataset=self.char_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.char_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _gen_CharacterEval(self, repeat_times=1):
        """ 验证 CharacterEval 数据集 """
        data_set = []

        for test_data in self.char_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)

            # 禁用 model-based RM 的 val
            if self.config.reward_model.enable and \
               test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # ==== 生成输出 ====
            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }

            # rollout generate
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch,
                self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded
            )
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded,
                pad_size=pad_size
            )

            # ==== 解码输出 ====
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]

            # merge
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info = test_batch.meta_info if hasattr(test_batch, 'meta_info') else {}
            test_batch.meta_info['validate'] = True
            test_batch.non_tensor_batch['output_text'] = output_texts

            data_set.append(test_batch)
            
        return data_set
        
        
    def _val_CharacterEval(self, data_set):
        from collections import defaultdict
        sample_scores = defaultdict(list)  # <-- 修改为按任务分类存储

        metric_dict = {}
        for test_batch in data_set:
            # ==== 调用 reward_fn ====
            grouped_scores = self.char_rm(test_batch, step=self.global_steps, has_thinking=self.thinking)

            # 分类收集
            for task_name, arr in grouped_scores.items():
                sample_scores[task_name].extend(arr)

        print("✅ CharacterEval 验证评分完成!")

        # ==== 输出分类验证指标 ====
        print("\n📊 分类验证结果：")
        for task_name, score_list in sample_scores.items():
            if len(score_list) == 0:
                continue

            arr = np.array(score_list, dtype=float)
            mean_v = arr.mean()

            wandb_key = f"Character_eval/{task_name}"
            metric_dict[wandb_key] = float(mean_v)

            # print(f"  {wandb_key}: mean={mean_v:.4f}, n={len(arr)}")

        key = [
            "Utterance",
            "Exposure",
            "Accuracy"
        ]

        for k in key:
            score_list = sample_scores[k]
            if len(score_list) == 0:
                continue

            arr = np.array(score_list, dtype=float)
            mean_v = arr.mean()

            wandb_key = f"val_chareval/{k}"
            metric_dict[wandb_key] = float(mean_v)
            
            print(f"  {wandb_key}: mean={mean_v:.4f}, n={len(arr)}")
        
        return metric_dict


    def _gen_rolellm(self):
        data_set = []
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            
            # 禁用 model-based RM 的 val
            if self.config.reward_model.enable and \
               test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # ==== 解码输入 ====
            input_ids = test_batch.batch['input_ids']
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]

            # ==== 生成输出 ====
            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }

            # rollout generate
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch,
                self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded
            )
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded,
                pad_size=pad_size
            )
            
            # merge
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info = test_batch.meta_info if hasattr(test_batch, 'meta_info') else {}
            test_batch.meta_info['validate'] = True

            data_set.append(test_batch)
            
        print("✅ Rolellm data generation done.")
        return data_set
        
    def _val_rolellm(self, data_set):
        from collections import defaultdict
        import numpy as np

        sample_scores = defaultdict(list)  # <-- 修改为按任务分类存储
        metric_dict = {}
        
        for test_batch in data_set:
            # ==== 调用 reward_fn ====
            grouped_scores = self.role_llm(test_batch, step=self.global_steps, has_thinking=self.thinking)  # <-- dict: task -> np.array
            
            for task_name, arr in grouped_scores.items():
                sample_scores[task_name].extend(arr)
            
        # 分类收集
        for task_name, arr in grouped_scores.items():
            sample_scores[task_name].extend(arr.tolist())

        # ==== 输出分类验证指标 ====
        print("\n📊 分类验证结果：")
        for task_name, score_list in sample_scores.items():
            if len(score_list) == 0:
                continue

            arr = np.array(score_list, dtype=float)
            mean_v = arr.mean()

            wandb_key = f"val_rolellm/{task_name}"
            metric_dict[wandb_key] = float(mean_v)

            print(f"  {wandb_key}: mean={mean_v:.4f}, n={len(arr)}")

        return metric_dict

    def _val_diversity(self):
        """ 执行多样性评估, 独立线程进行并在完成后记录 """
        from collections import defaultdict
        import numpy as np

        sample_scores = defaultdict(list) 
        metric_dict = {}
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            
            test_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                                                        dtype=object)
            
            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True
            )

            # 禁用 model-based RM 的 val
            if self.config.reward_model.enable and \
               test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # ==== 生成输出 ====
            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }

            # rollout generate
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch,
                self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded
            )
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded,
                pad_size=pad_size
            )
            print("✅ Diversity data generation done.")

            # merge
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info = test_batch.meta_info if hasattr(test_batch, 'meta_info') else {}
            test_batch.meta_info['validate'] = True
        
            scores = self.diversity(data=test_batch, step=self.global_steps, has_thinking=self.thinking)
        
            for task_name, arr in scores.items():
                sample_scores[task_name].extend(arr.tolist())
            
        # ==== 输出分类验证指标 ====
        print("\n📊 分类验证结果：")
        for task_name, score_list in sample_scores.items():
            if len(score_list) == 0:
                continue

            arr = np.array(score_list, dtype=float)
            mean_v = arr.mean()

            wandb_key = f"Diversity/{task_name}"
            metric_dict[wandb_key] = float(mean_v)

            # print(f"  {wandb_key}: mean={mean_v:.4f}, n={len(arr)}")

        key = [
            "distinct-2",
            "rouge-L",
            "chamfer_dist",
            "semantic_entropy",
            "num_clusters"
        ]

        for k in key:
            score_list = sample_scores[k]
            if len(score_list) == 0:
                continue

            arr = np.array(score_list, dtype=float)
            mean_v = arr.mean()

            wandb_key = f"val_diversity/{k}"
            metric_dict[wandb_key] = float(mean_v)
            
            print(f"  {wandb_key}: mean={mean_v:.4f}, n={len(arr)}")

        return metric_dict


    def _validate(self):
        data_set = self._gen_CharacterEval()
        test_batch = self._gen_rolellm()
        
        print("✅ CharacterEval 验证生成完成!")
        
        res_rolellm = {}
        res_char_rm = {}
        res_diversity = {} 
        
        # 2. 定义两个并行线程：一个评估 CharacterEval，一个跑_validate
        t1 = threading.Thread(target=lambda: res_rolellm.setdefault("res", self._val_rolellm(test_batch)))  # 包含本地推理 + 评估
        t2 = threading.Thread(target=lambda: res_char_rm.setdefault("res", self._val_CharacterEval(data_set)))
        t3 = threading.Thread(target=lambda: res_diversity.setdefault("res", self._val_diversity()))

        # 3. 启动并行
        t1.start()
        t2.start()
        t3.start()

        # 4. 等待结束
        t1.join()
        t2.join()
        t3.join()
        
        metrics = res_rolellm["res"]
        metrics.update(res_char_rm["res"])
        metrics.update(res_diversity["res"])
        
        return metrics


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:                 # False
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs        未实现
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            # 读取最后保存的ckpt, 返回其路径
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':   # default True
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:               # 默认调过,即手动设置本地路径在 trainer.resume_mode 时才执行
            # resume_from_path 默认 false 所以只取决于是否找到了本地路径
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                # 确保本地路径包含 global_step_ 字样
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):   # 不是绝对路径时进行处理
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()         # 支持断点重续，读取最后保存的模型文件
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            self.print(f'Initial validation metrics: {val_metrics}')
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        start_time = time()
        length = len(self.train_dataloader)

        for epoch in range(self.config.trainer.total_epochs):
            for i, batch_dict in enumerate(self.train_dataloader):
                def format_seconds(s):
                    return f"{int(s // 3600):02d}h{int((s % 3600) // 60):02d}m{int(s % 60):02d}s"
                use_time = time() - start_time
                if i == 0:
                    infor = "Training begin!"
                else:
                    infor = "   epoch: {}\tstep: {}/{}\tglobal_steps: {}\ttime: {}/{}   ".format(
                    epoch, i, length, self.global_steps, format_seconds((use_time/self.global_steps)*i), format_seconds((use_time/self.global_steps)*length))
                print("=" * 20 + infor + "=" * 20)
                self.reward_fn.print("=" * 20 + infor + "=" * 20)
                
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)   # : 作用限定 batch 类型

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():   # 多模态模型, False
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )
                is_last_step = self.global_steps >= self.total_training_steps
                
                # self.actor_rollout_wg用于生成轨迹数据（生成token，奖励）执行被训模型向前推理
                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        # 调用本地被训练模型，获得回复，这里生成的已经是复制同一输入多个输出了
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # TODO: REMAX优势估计器，需要再学习
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:     # False
                        with _timer('gen_max', timing_raw):
                            # 针对训练样本，调用被训模型重新生成一遍轨迹
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            # 使用奖励模型计算每个step的reward
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            # 沿时间维度求和，得到每个轨迹（生成句子）的总奖励
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            # 去除临时合并的基准轨迹数据（恢复回原始batch）
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            # 记录总奖励值
                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    # 创建uid字段，为配对做准备
                    # 对于原始的prompt batch，我们为每个prompt分配一个唯一的uid
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    
                    # repeat to align with repeated responses in rollout
                    # 复制batch以匹配数据并行，之后进行合并, rollout.n 默认 8
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # 平衡并行处理的token数量，均匀分布
                    if self.config.trainer.balance_batch:       # True
                        self._balance_batch(batch, metrics=metrics)

                    # 统计已计算过的token数量
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    # 计算旧策略的动作概率
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:       # 计算参考的动作概率                True
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    # compute values
                    if self.use_critic:                 # 使用critic网络计算状态价值V       False
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:                 # 计算奖励模型分数                  False
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        # 现在配对逻辑已经迁移到naive文件中，这里直接调用reward_fn
                        try:
                            # reward_tensor是与batch[response](tonken化)相同尺度的奖励张量，原句子最后一个有效token位置设置为得分
                            # batch_new_samples 是空列表 []
                            # thinking 决定是否令模型执行thinking推理
                            if 'prompt_str' not in batch[0].non_tensor_batch['extra_info'].keys():
                                for data_item in batch:
                                    prompt_ids = data_item.batch['prompts']
                                    prompt_length = prompt_ids.shape[-1]
                                    valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                                    valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                                    data_item.non_tensor_batch['extra_info']['prompt_str'] = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                            
                            reward_tensor, thinking_tensor, diversity_tensor = self.reward_fn(batch, self.thinking)    # 调用vllm服务进行打分
                        except Exception as e:
                            print(f"Warning: reward_fn failed, using default: {e}")
                            # 如果reward_fn失败，使用默认值
                            reward_tensor = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)
                            thinking_tensor = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)
                            diversity_tensor = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)

                        batch.batch['token_level_scores'] = reward_tensor + thinking_tensor + diversity_tensor * self.config.trainer.diversity_ratio
                        batch.batch['reward_scores'] = reward_tensor
                        batch.batch['thinking_scores'] = thinking_tensor
                        batch.batch['diversity_scores'] = diversity_tensor

                        # compute rewards. apply_kl_penalty if available
                        # 计算kl惩罚，即不希望rl模型与原始模型区别过大
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):   # True
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                    # update critic
                    if self.use_critic:             # 优化（训练）Critic网络        False
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    # 仅在critic优化足够多的步数后才对Actor（被训模型）进行优化，避免早期的不稳定
                    if self.config.trainer.critic_warmup <= self.global_steps:      # critic_warmup = 0, 故稳定进入
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        # 在 gpu 维度做平均，从而获得并行计算后的最终结果
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    # 统计奖励值，用于 wandb 记录
                    reward_metrics = compute_reward_metrics(batch)
                    metrics.update(reward_metrics)

                    # validate
                    if self.config.trainer.test_freq > 0 and \
                        (is_last_step or self.global_steps % self.config.trainer.test_freq == 0) and \
                            self.config.trainer.critic_warmup <= self.global_steps:
                        print("=" * 50 + 'validate begin!' + "=" * 50) 
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        
                        metrics["val_diversity/entropy_loss"] = metrics["actor/entropy_loss"]
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0) and \
                                self.config.trainer.critic_warmup <= self.global_steps:
                        print("=" * 50 + f'save_checkpoint {self.global_steps}' + "=" * 50)
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                self.logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return

                self.global_steps += 1
                
                