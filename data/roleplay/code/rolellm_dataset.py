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
角色扮演数据转换脚本

输入格式 (JSONL):
{
    "role": "孙悟空", 
    "question": "编辑以下句子中的语法错误：他们去得机场太晚，结果错过了飞机。", 
    "generated": ["哈哈飞机。"]
}

角色描述文件 (desc.json):
{
    "孙悟空": "西游记中的主角之一，齐天大圣，性格顽皮机智，武艺高强，火眼金睛能识破妖魔鬼怪。说话直率，有时略显傲慢，但内心正义，保护师父唐僧西天取经。",
    ...
}

输出格式 (Parquet):
{
    "data_source": "roleplay",
    "prompt": [{"role": "user", "content": "完整提示词+用户问题"}],
    "ability": "roleplay",
    "extra_info": {
        "split": "train",
        "index": 0,
        "role_name": "孙悟空",
        "first_category": "Character"
    }
}
"""

import os
import json
import random
import pandas as pd
from typing import List, Dict, Any


SYSTEM_THINK_CN_O = """
你将始终扮演：“{role_name}”。
你的角色描述为：“{role_description}”。

接下来你的每一次回复必须严格采用以下结构输出：

<think>
这里写出你的思考过程、推理过程、分析步骤。
内容必须是自然的思考，目的是更好地进行角色扮演。
</think>

<answer>
这里给出你身为 {role_name} 给出的的最终回复。
内容必须简洁、明确，符合角色。
</answer>

**禁止输出除此之外的任何额外内容。**  
**如果你严格按照格式输出，将会获得奖励。**

你在<answer>和</answer>之间的输出将被视为你的最终回答，必须完全符合该角色的语气、认知、习惯与表达方式，不得跳出角色。
你的目标是让用户通过你的回答感受到你就是 **{role_name}**。
"""

SYSTEM_THINK_EN_O = """
You are about to assume the role: "{role_name}".
Your role description is: "{role_description}".

From now on, every response you produce must strictly follow the structure below:

<think>
Write your thinking process, reasoning process, and analysis steps here.
The content must reflect natural thinking, with the goal of better performing the role-play.
</think>

<answer>
Provide the final response as "{role_name}" here.
The content must be concise, clear, and consistent with the role.
</answer>

**Do not output any additional content beyond what is specified above.**
**If you strictly follow the required format, you will receive a reward.**

The content you output between <answer> and </answer> will be considered your final answer.
It must fully conform to the role’s tone, knowledge, habits, and expression style, and you must not break character.

Your goal is to make the user clearly feel that you are **"{role_name}"** through your response.
"""


SYSTEM_THINK_CN = """
你将始终扮演：“{role_name}”。
你的角色描述为：“{role_description}”。

接下来你的每一次回复必须严格采用以下结构输出：


<think>
这里写出你的思考过程、推理过程、分析步骤。内容必须是自然的思考，目的是更好地进行角色扮演。
</think>

这里给出你身为 {role_name} 给出的的第一人称回复。内容必须简洁、明确，符合角色。


**如果你严格按照格式输出，将会获得奖励。**
你在</think>之后的输出将被视为你的最终回答，必须完全符合该角色的语气、认知、习惯与表达方式，不得跳出角色。
你的目标是让用户通过你的回答认为你就是 **{role_name}**。
"""


SYSTEM_THINK_EN = """
You are about to begin role-playing as: "{role_name}".
Your role description is: "{role_description}".

From now on, every response you generate must strictly follow the structure below:

<think>
Write your chain of thought here, including your reasoning process, inference steps, and analysis.  
The content must reflect natural thinking, with the goal of better performing the role-play.
</think>

Here, provide the first-person response as {role_name}.  
The content must be concise, clear, and fully aligned with the role.

**If you strictly follow the required format, you will receive a reward.**

Any content you output after </think> will be treated as your final answer.  
It must completely conform to the role’s tone, cognition, habits, and style of expression.  
You must not break character.

Your objective is to make the user believe, through your response, that you truly are **{role_name}**.
"""


SYSTEM_NOTHINK_CN = """
你将始终扮演：“{role_name}”。
你的角色描述为：“{role_description}”。

你的回答，必须完全符合该角色的语气、认知、习惯与表达方式，不得跳出角色。
你的目标是让用户通过你的回答认为你就是 **{role_name}**。
"""

SYSTEM_NOTHINK_EN = """
You are about to assume the role: "{role_name}".
Your role description is: "{role_description}".

Your answer must fully conform to the role’s tone, knowledge, habits, and expression style, and you must not break character.
Your goal is to make the user clearly feel that you are **"{role_name}"** through your response.
"""


def load_role_descriptions(desc_file_path: str) -> Dict[str, str]:
    """
    从desc.json文件加载角色描述
    
    Args:
        desc_file_path: 角色描述文件路径
    
    Returns:
        角色名称到描述的映射字典
    """
    try:
        with open(desc_file_path, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
        print(f"成功加载 {len(descriptions)} 个角色描述")
        return descriptions
    except FileNotFoundError:
        print(f"警告: 找不到角色描述文件 {desc_file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"错误: 解析角色描述文件失败 - {e}")
        return {}


def create_system_prompt(role_name: str, role_description: str, language: str = "cn") -> str:
    """
    创建角色扮演的系统提示词
    
    Args:
        role_name: 角色名称
        role_description: 角色描述
    
    Returns:
        系统提示词字符串
    """
    
    system_prompt_cn = (SYSTEM_NOTHINK_CN.format(role_name=role_name, role_description=role_description))
    
    system_prompt_en = (SYSTEM_NOTHINK_EN.format(role_name=role_name, role_description=role_description))
    
    system_prompt = {
        "cn": system_prompt_cn,
        "en": system_prompt_en
    }
    
    return system_prompt[language]


def create_model_prefix(model_type: str, system_content: str, user_question: str) -> str:
    """
    根据不同模型类型创建对话模板
    
    Args:
        model_type: 模型类型 ('qwen3', 'llama3', 'chatglm', 'baichuan', 'general')
        system_content: 系统提示词内容
        user_question: 用户问题
    
    Returns:
        格式化的对话模板字符串
    """
    
    if model_type.lower() == 'qwen3':
        # Qwen3 模板
        prefix = f"""<|im_start|>system
{system_content}
<|im_end|>
<|im_start|>user
{user_question}
<|im_end|>
<|im_start|>assistant
"""
        
    elif model_type.lower() == 'llama3':
        # Llama3 模板
        prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
    elif model_type.lower() == 'chatglm':
        # ChatGLM 模板
        prefix = f"""[gMASK]<sop><|system|>
{system_content}
<|user|>
{user_question}
<|assistant|>
"""
        
    elif model_type.lower() == 'baichuan':
        # Baichuan 模板
        prefix = f"""<reserved_106>{system_content}<reserved_107>{user_question}<reserved_108>"""
        
    elif model_type.lower() == 'general':
        # 通用模板（类似ChatML格式）
        prefix = f"""<|system|>
{system_content}
<|user|>
{user_question}
<|assistant|>
"""
        
    else:
        # 默认使用Qwen3模板
        print(f"警告: 未知模型类型 '{model_type}'，使用Qwen3模板")
        prefix = f"""<|im_start|>system
{system_content}
<|im_end|>
<|im_start|>user
{user_question}
<|im_end|>
<|im_start|>assistant
"""
    
    return prefix


def read_character_jsonl_data(file_path: str, role_descriptions: Dict[str, str], model_type: str = 'qwen3', source: str = 'roleplay', task_name: str = "train") -> List[Dict[str, Any]]:
    """
    读取新格式的角色扮演JSONL数据
    
    Args:
        file_path: JSONL文件路径
        role_descriptions: 角色描述映射
        model_type: 模型类型，用于选择对话模板
        source: 数据来源标识
    
    Returns:
        处理后的数据列表
    """
    all_data = []
    missing_descriptions = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                
                # 提取字段信息
                role_name = data.get('role', '')
                question = data.get('question', '')
                language = data.get('language', '')
                
                # 验证必要字段
                if not role_name or not question:
                    print(f"警告: 第 {line_num + 1} 行缺少必要字段，跳过...")
                    continue
                
                # 获取角色描述
                role_description = role_descriptions.get(role_name, '')
                if not role_description:
                    missing_descriptions.add(role_name)
                    # 使用默认描述
                    role_description = f"一个名为{role_name}的角色"
                
                # 生成系统提示词
                system_prompt = create_system_prompt(role_name, role_description, language)
                
                # 生成完整的模型特定格式
                if language == "cn":
                    full_prompt = create_model_prefix(model_type, system_prompt, "我的问题是：" + question)
                elif language == "en":
                    full_prompt = create_model_prefix(model_type, system_prompt, "My question is: " + question)
                else:
                    raise KeyError("错误")
                
                # 构建训练数据
                training_item = {
                    "data_source": source,
                    "prompt": [{
                        "role": "user",
                        "content": full_prompt
                    }],
                    "ability": "roleplay",
                    "extra_info": {
                        'split': 'test',
                        'index': len(all_data),
                        'role_name': role_name,
                        'role_desc': role_description,
                        'user_question': question,
                        'first_category': 'Character',
                        'model_type': model_type,
                        "task_name": task_name,
                        "language": language
                    }
                }
                
                all_data.append(training_item)
                
            except json.JSONDecodeError as e:
                print(f"错误: 解析第 {line_num + 1} 行失败 - {e}")
                continue
            except Exception as e:
                print(f"意外错误在第 {line_num + 1} 行: {e}")
                continue
    
    # 报告缺失描述的角色
    if missing_descriptions:
        print(f"警告: 以下角色缺少描述信息: {', '.join(missing_descriptions)}")
    
    return all_data



def main():
    """主函数：处理数据转换流程"""
    task_name = "raw"
    # 配置路径
    save_dir = f"/gemini/space/private/cgn/project/cllm_rl/data/roleplay/{task_name}.parquet"   # 输出目录
    input_file = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/tem/test_raw.jsonl" 
    desc_file = "/gemini/space/private/cgn/project/cllm_rl/data/roleplay/tem/desc.json" 
    
    # 配置模型类型 - 可选择: 'qwen3', 'llama3', 'chatglm', 'baichuan', 'general'
    model_type = 'qwen3'  # 默认使用Qwen3模板，可根据需要修改
    
    # 数据来源标识
    data_source = 'roleplay'
    
    print(f"开始处理角色扮演数据...")
    print(f"输入文件: {input_file}")
    print(f"角色描述文件: {desc_file}")
    print(f"输出目录: {save_dir}")
    print(f"模型类型: {model_type}")
    
    # 加载角色描述
    role_descriptions = load_role_descriptions(desc_file)
    
    # 读取原始数据
    dataset = read_character_jsonl_data(input_file, role_descriptions, model_type, data_source, task_name)
    print(f"成功读取 {len(dataset)} 条数据")
    
    if len(dataset) == 0:
        print("错误: 没有有效的数据，请检查输入文件格式")
        return
    
    # 显示示例数据
    if dataset:
        print("\n数据示例:")
        print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(dataset)
    
    # 创建输出目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建输出目录: {save_dir}")
    
    
    # 保存为Parquet格式
    output_path = os.path.join(save_dir, f'{model_type}_rolellm_train.parquet')
    df.to_parquet(output_path, index=False)
    """
    
    df.to_parquet(save_dir, index=False)
    
    print(f"\n数据转换完成！")
    print(f"输出文件: {save_dir}")
    
    # 显示数据统计信息
    print(f"\n数据统计:")
    print(f"总数据量: {len(dataset)}")
    print(f"模型类型: {model_type}")
    
    # 显示角色分布
    role_counts = {}
    for item in dataset:
        role_name = item['extra_info']['role_name']
        role_counts[role_name] = role_counts.get(role_name, 0) + 1
    
    # print(f"\n角色分布:")
    # for role_name, count in sorted(role_counts.items()):
        # print(f"  {role_name}: {count}")
    
    # 显示可用的模型类型
    print(f"\n支持的模型类型:")
    print("  - qwen3: Qwen3模板")
    print("  - llama3: Llama3模板") 
    print("  - chatglm: ChatGLM模板")
    print("  - baichuan: Baichuan模板")
    print("  - general: 通用ChatML模板")
    print(f"\n如需使用其他模型模板，请修改main()函数中的model_type变量")


if __name__ == '__main__':
    main()

'''
python /gemini/space/private/lyh/Code/role_grpo/examples/data_preprocess/rolellm_dataset.py

'''