# Roleplay 奖励函数使用指南

本指南介绍如何在verl框架中使用 `roleplay.py` 奖励函数进行角色扮演训练。

## 🎯 核心设计理念

### 1. 两两配对比较机制

- **rollout=16**: 每个用户查询生成16个回复
- **配对策略**: 16个回复按顺序两两配对，形成8个比较对
- **比较逻辑**: 每个回复都与配对的另一个回复进行比较评估

### 2. 奖励计算流程

```
用户查询 → 生成16个回复 → 两两配对 → 比较评估 → 计算奖励
```

## 🔧 奖励函数接口

### 函数签名

```python
def compute_score(data_source: str, solution_str: str, ground_truth: str, 
                  extra_info: Dict[str, Any] = None) -> float:
```

### 参数说明

- `data_source`: 数据来源标识（"roleplay"）
- `solution_str`: 当前需要评估的回复
- `ground_truth`: 标准答案（在roleplay场景下不使用）
- `extra_info`: 额外信息，包含配对回复和角色信息

### extra_info 结构

```python
extra_info = {
    'paired_response': '配对的另一个回复',
    'role_name': '角色名称',
    'first_category': '角色类别',
    'prompt_str': '完整的提示词'
}
```

## 🚀 训练流程集成

### 1. 在ray_trainer.py中配置

```python
# 设置rollout次数为16
config.generation.num_return_sequences = 16

# 在生成batch输出后，进行配对处理
def process_gen_batch_output(gen_batch_output, role_name, first_category, prompt_str):
    # 将16个回复按顺序两两配对
    paired_responses = []
    for i in range(0, len(gen_batch_output), 2):
        if i + 1 < len(gen_batch_output):
            paired_responses.append({
                'response_a': gen_batch_output[i],
                'response_b': gen_batch_output[i + 1]
            })
    
    return paired_responses
```

### 2. 在DataProto中添加配对信息

```python
# 将配对信息添加到non_tensor_batch中
def add_pairing_info_to_dataproto(data_proto, paired_responses):
    # 确保non_tensor_batch存在
    if not hasattr(data_proto, 'non_tensor_batch'):
        data_proto.non_tensor_batch = {}
    
    # 为每个配对添加信息
    for i, pair in enumerate(paired_responses):
        # 为response_a设置配对信息
        if i * 2 < len(data_proto):
            if 'extra_info' not in data_proto.non_tensor_batch:
                data_proto.non_tensor_batch['extra_info'] = [{}] * len(data_proto)
            
            data_proto.non_tensor_batch['extra_info'][i * 2].update({
                'paired_response': pair['response_b'],
                'role_name': pair['role_name'],
                'first_category': pair['first_category'],
                'prompt_str': pair['prompt_str']
            })
        
        # 为response_b设置配对信息
        if i * 2 + 1 < len(data_proto):
            if 'extra_info' not in data_proto.non_tensor_batch:
                data_proto.non_tensor_batch['extra_info'] = [{}] * len(data_proto)
            
            data_proto.non_tensor_batch['extra_info'][i * 2 + 1].update({
                'paired_response': pair['response_a'],
                'role_name': pair['role_name'],
                'first_category': pair['first_category'],
                'prompt_str': pair['prompt_str']
            })
    
    return data_proto
```

### 3. 在NaiveRewardManager中使用

```python
# 在verl/workers/reward_manager/naive.py中
from verl.utils.reward_score.roleplay import compute_score

# 设置compute_score函数
def setup_roleplay_reward_manager(tokenizer):
    from verl.workers.reward_manager import NaiveRewardManager
    
    reward_fn = NaiveRewardManager(
        tokenizer=tokenizer, 
        num_examine=0, 
        compute_score=compute_score  # 使用roleplay奖励函数
    )
    
    return reward_fn
```

## 🔧 具体实现步骤

### 步骤1: 修改ray_trainer.py

在 `verl/trainer/ppo/ray_trainer.py` 的 `fit` 方法中，找到生成batch输出的位置：

```python
# 生成batch输出
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

# 在这里添加roleplay配对处理
if batch.non_tensor_batch.get('data_source', [''])[0] == 'roleplay':
    # 提取角色信息
    extra_info = batch.non_tensor_batch.get('extra_info', [{}])
    if extra_info and len(extra_info) > 0:
        role_name = extra_info[0].get('role_name', '')
        first_category = extra_info[0].get('first_category', '')
        prompt_str = extra_info[0].get('prompt_str', '')
        
        # 处理配对
        paired_responses = process_gen_batch_output(
            gen_batch_output, 
            role_name, 
            first_category, 
            prompt_str
        )
        
        # 将配对信息添加到DataProto
        batch = add_pairing_info_to_dataproto(batch, paired_responses)

# 继续原有流程
batch = batch.union(gen_batch_output)
```

### 步骤2: 配置rollout次数

确保配置文件中设置正确的rollout次数：

```yaml
# config/generation.yaml
generation:
  num_return_sequences: 16  # 每个查询生成16个回复
  temperature: 0.8
  top_p: 0.9
  max_new_tokens: 512
```

### 步骤3: 设置奖励函数

在 `main_ppo.py` 中导入并使用roleplay奖励函数：

```python
# 在verl/trainer/main_ppo.py中
from verl.utils.reward_score.roleplay import compute_score

# 创建奖励管理器
reward_fn = NaiveRewardManager(
    tokenizer=tokenizer, 
    num_examine=0, 
    compute_score=compute_score  # 使用roleplay奖励函数
)
```

## 📊 奖励计算逻辑

### 1. 基础评估维度

- **结构验证** (20%): 检查回复格式和AI标识
- **角色一致性** (40%): 评估是否保持角色身份
- **专业领域** (20%): 根据first_category进行专业评估
- **完整性** (10%): 回复长度和内容完整性
- **语言风格** (10%): 符合角色特点的语言风格

### 2. 类别特定评估

```python
# 政府类别
if first_category == 'Government':
    if '公文' in response or '政策' in response or '行政' in response:
        score += 0.2

# 法律类别
elif first_category == 'Law':
    if '法律' in response or '法规' in response or '条款' in response:
        score += 0.2

# 医疗类别
elif first_category == 'Medical':
    if '医疗' in response or '健康' in response or '治疗' in response:
        score += 0.2
```

### 3. 自定义比较逻辑

在 `compare_responses` 函数中，你可以实现自己的比较逻辑：

```python
def compare_responses(response_a: str, response_b: str, first_category: str, 
                     role_info: Dict[str, str]) -> Tuple[float, float]:
    # TODO: 在这里实现你的比较逻辑
    # 这里只是一个示例，你需要根据实际需求实现
    
    # 你的自定义评估标准
    score_a = your_evaluation_logic(response_a, first_category, role_info)
    score_b = your_evaluation_logic(response_b, first_category, role_info)
    
    return score_a, score_b
```

## 🔄 训练循环示例

### 完整的训练流程

```python
# 1. 生成16个回复
output = actor_rollout_ref_wg.generate_sequences(prompt)

# 2. 处理生成输出，进行配对
paired_outputs = process_gen_batch_output(output, role_name, first_category, prompt_str)

# 3. 将配对信息添加到DataProto
batch = add_pairing_info_to_dataproto(batch, paired_outputs)

# 4. 计算奖励
rewards = reward_fn(batch)  # 调用roleplay奖励函数

# 5. 继续PPO训练流程
advantages = compute_advantages(values, rewards)
actor_rollout_ref_wg.update_actor(output)
critic.update_critic(output)
```

## ⚙️ 配置参数

### 1. 生成配置

```yaml
# config/generation.yaml
generation:
  num_return_sequences: 16  # 每个查询生成16个回复
  temperature: 0.8
  top_p: 0.9
  max_new_tokens: 512
```

### 2. 训练配置

```yaml
# config/ppo_trainer.yaml
trainer:
  rollout_batch_size: 8  # 确保能被16整除
  ppo_epochs: 4
  learning_rate: 1e-5
```

## 🧪 测试和调试

### 1. 测试奖励函数

```bash
cd verl/utils/reward_score
python roleplay.py
```

### 2. 验证配对逻辑

```python
# 测试配对是否正确
test_responses = [f"response_{i}" for i in range(16)]
paired = process_gen_batch_output(test_responses, "测试角色", "测试类别", "测试提示词")
print(f"生成了 {len(paired)} 个配对")
```

### 3. 检查奖励分布

```python
# 分析奖励分数分布
rewards = reward_fn(data_proto)
print(f"奖励分数范围: {rewards.min():.3f} - {rewards.max():.3f}")
print(f"平均奖励: {rewards.mean():.3f}")
```

## ⚠️ 注意事项

1. **rollout数量**: 必须设置为16，确保能形成8个配对
2. **配对顺序**: 按生成顺序两两配对，确保一致性
3. **extra_info完整性**: 必须包含所有必要的配对信息
4. **类别评估**: 根据first_category实现相应的专业评估标准
5. **内存管理**: 16个回复会占用更多内存，注意资源分配

## 🔗 相关文件

- `verl/utils/reward_score/roleplay.py`: 奖励计算函数
- `verl/utils/reward_score/roleplay_ray_integration.py`: Ray训练器集成示例
- `verl/workers/reward_manager/naive.py`: 奖励管理器
- `verl/trainer/ppo/ray_trainer.py`: PPO训练器
- `examples/data_preprocess/roleplay_dataset.py`: 数据预处理脚本

## 📞 技术支持

如果在集成过程中遇到问题：

1. 检查rollout配置是否正确设置为16
2. 验证配对逻辑是否正确实现
3. 确认extra_info包含所有必要信息
4. 检查奖励函数的返回值范围

更多技术细节请参考verl框架的官方文档。
