import os
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import gsm8k, math_verify, table, MATH, normal_reasoning, table_critic
from verl.utils.reward_score import roleplay
from verl.utils.logger.logger import Logger
import ray
import hydra

import nltk
nltk.download = lambda *args, **kwargs: None

def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval', 'MATH']:
        return MATH.compute_score
    elif data_source in ['normal_reasoning', 'orz_math', 'olympiad_bench', 'AMC', 'math500', 'AIME']:
        return normal_reasoning.compute_score
    elif data_source in [
        'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
        'numina_olympiads']:
        from verl.utils.reward_score import prime_math
        return prime_math.compute_score
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from verl.utils.reward_score import prime_code
        return prime_code.compute_score
    elif data_source in ['hiyouga/geometry3k']:
        from verl.utils.reward_score import geo3k
        return geo3k.compute_score
    elif data_source in ['table', 'table-190', 'table_rethink']:
        return table.compute_score
    elif data_source in ['table_critic', 'table_compare']:
        return table_critic.compute_score
    elif data_source in ['roleplay']: 
        return roleplay.compute_score
    else:
        raise NotImplementedError


def compute_score(data_source, solution_str, ground_truth, prompt_str, extra_info=None):
    print("\n" + "=" * 80)
    print(" Processing New Sample ".center(80, '='))
    print("Data Source: {}".format(data_source))

    if data_source == 'table_compare':
        print(f"\n[Model Input]\n{prompt_str}")
    if data_source == "table_rethink" or data_source == 'table_critic':
        if '<|im_start|>assistant' in prompt_str:
            print(f"\n[Previous Model Response]\n{'<|im_start|>assistant'.join(prompt_str.split('<|im_start|>assistant')[1:])}")
        elif "<_bot>" in prompt_str:
            print(
                f"\n[Previous Model Response]\n{'<_bot>'.join(prompt_str.split('<_bot>')[1:])}")
        elif '<｜Assistant｜>' in prompt_str:
            print(
                f"\n[Previous Model Response]\n{'<｜Assistant｜>'.join(prompt_str.split('<｜Assistant｜>')[1:])}")

    print(f"\n[Model Response]\n{solution_str}")
    print("********")
    print(f"\n[Ground Truth]\n{ground_truth}")

    compute_score_fn = _select_rm_score_fn(data_source)
    if data_source == 'roleplay':
        # roleplay需要extra_info参数
        res, new_samples = compute_score_fn(data_source, solution_str, ground_truth, extra_info=extra_info)
    elif 'table' in data_source.lower():
        res, new_samples = compute_score_fn(solution_str, ground_truth, prompt_str, extra_info=extra_info)
    else:
        res, new_samples = compute_score_fn(solution_str, ground_truth, prompt_str)

    print(f"  Total Score: {res}")
    print("=" * 80 + "\n")

    if isinstance(res, (int, float, bool)):
        return float(res), new_samples
    else:
        return float(res[0]), new_samples


def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


# 加载 config/ppo_trainer.yaml      命令行输入优先覆盖
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',   # 允许tokenizer使用多线程
                'NCCL_DEBUG': 'WARN',               # NCCL库只输出warning
                'VLLM_LOGGING_LEVEL': 'WARN'        # 降低VLLM日志输出级别
            }
        })

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)  # 该任务需求一个cpu
def main_task(config):
    from verl.utils.fs import copy_to_local
    from pprint import pprint
    from omegaconf import OmegaConf
    
    has_thinking = config.has_thinking
    del config.has_thinking
    print("="*20 + f"has_thinking = {has_thinking}" + "="*20)
    
    # 将conf转化为python结构并打印，resolve代表将插值修改为对应值
    pprint(OmegaConf.to_container(config, resolve=True))  
    OmegaConf.resolve(config)           # 显式解析插值并修改，例如： PATH -> 具体路径

    # 如果是HDFS路径，下载并返回缓存路径，否则不做处理
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # 初始化 tokenizer 和 processor(多模态模型中用于图像/文本混合输入)
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # None

    # 选择 worker 组管理类
    if config.actor_rollout_ref.actor.strategy == 'fsdp':       # default
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # 构建映射
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),   # 被训练模型
        Role.Critic: ray.remote(CriticWorker),                  # 评估模型
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)       # 参考模型, 不使用
    }

    # 两个资源池：
    # actor/ref 用前4张卡
    # critic 用后4张卡
    actor_ref_pool_id = "actor_ref_pool"
    critic_pool_id = "critic_pool"

    resource_pool_spec = {
        actor_ref_pool_id: [2],   # 单机8卡时，给 actor+ref 4 张卡
        critic_pool_id: [2],      # 单机8卡时，给 critic 4 张卡
    }

    # 角色到资源池的映射
    mapping = {
        Role.ActorRollout: actor_ref_pool_id,
        Role.RefPolicy: actor_ref_pool_id,
        Role.Critic: critic_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:      # default: False
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id
    
    """
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':      # default
        from verl.workers.reward_manager import ThinkRewardManager
        reward_manager_cls = ThinkRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError
    """
    
    from verl.workers.reward_manager import PPORewardManager, RolellmRewardManager, CharacterRewardManager, DiversityRewardManager
    from config import CHAR_API_BASE, EMBEDDING_API_BASE
    
    # 奖励计算器，调用vllm提供的模型进行分数计算
    naive_log = Logger("naive")
    reward_fn = PPORewardManager(tokenizer=tokenizer, config=config, p_function=naive_log.log)

    # RoleLLM 验证集, 分别计算 cus, spe, raw 得分
    role_llm = RolellmRewardManager(tokenizer=tokenizer, num_examine=0,config=config)
    
    # CharacterEval 验证集
    char_rm = CharacterRewardManager(url=CHAR_API_BASE)
    
    # 语义熵评估
    diversity = DiversityRewardManager(
                                    tokenizer=tokenizer,
                                    embedding_url=EMBEDDING_API_BASE,
                                    config=config
                                )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    
    # TODO: 日志输出记录方法
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            processor=processor,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            role_llm=role_llm,
                            char_rm = char_rm,
                            diversity = diversity,
                            has_thinking=has_thinking)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
