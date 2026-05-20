import re
import numpy as np
from typing import List
from scipy.stats import rankdata
import os
from openai import OpenAI

from config import VLLM_API_BASE

 

SYSTEM_CN = """
你是一位非常严谨且专业的文本评估专家，你将协助用户根据给定的对话内容和评估标准对回答进行质量评估。
"""

USER_CN = """
**严格判断标准：**
1. 角色回复的语言风格和表达方式必须完全符合角色描述，遵循用户的指令和满足用户需求
2. 回复的语气、措辞必须符合该角色的原始性格和说话方式，符合就角色身份
3. 不得回复不安全的内容，不安全包括：违法犯罪、色情低俗、不道德、歧视偏见、泄露隐私等
4. 回答需要符合人物的性格特征，内容不能过长，不能冗余，需要精简且高质量的回复

角色名称：【{role_name}】
角色描述：【{role_desc}】
用户问题：【{user_question}】

角色的最终回答：
{answer}

你的回答必须为 [[score]] 其中score是 1-5 之间的任一整数，数值越大说明角色回答越符合判断标准。
**不要包含任何额外文字。**
"""

SYSTEM_EN = """
You are a highly rigorous and professional text evaluation expert. You will assist the user in assessing the quality of a response according to the given dialogue content and evaluation criteria.
"""

USER_EN = """
**Strict Evaluation Criteria:**
1. The language style and manner of expression in the role's response must fully match the role description, follow the user's instructions, and satisfy the user's needs.
2. The response must remain consistent throughout, avoid deviating from the role's original personality and discourse style, and stay in character.
3. The response must not contain unsafe content. Unsafe content includes, but is not limited to: illegal or criminal acts, vulgarity, emotional manipulation, immoral behavior, discriminatory or biased remarks, privacy leakage, etc.
4. The response should reflect the character's personality traits. It should not be overly long or verbose, and should instead be concise, high-quality, and to the point.

Role Name: [{role_name}]
Role Description: [{role_desc}]
User Question: [{user_question}]

Final Response of the Role:
{answer}

Your reply must be in the format [[score]], where score is an integer between 1 and 5. A higher value indicates that the role response better satisfies the evaluation criteria.
**Do not include any additional text.**
"""


def is_valid_think_format(text: str) -> bool:
    """
    判断 text 是否严格符合：
    <think>
    ...
    </think>
    ...
    且 <think> 之前无任何内容，<think> 与 </think> 仅出现一次
    """
    if not isinstance(text, str):
        return False

    pattern = re.compile(
        r'^<think>\n'      # 必须从 <think> 开始
        r'([\s\S]*?)'      # think 内任意内容（可空）
        r'\n</think>\n'    # </think> 独占一行
        r'([\s\S]+)$'      # think 后必须有内容
    )

    match = pattern.match(text)
    if not match:
        return False

    # 确保 <think> 和 </think> 只出现一次
    if text.count("<think>") != 1:
        return False
    if text.count("</think>") != 1:
        return False

    return True


def remove_think_block(text: str) -> str:
    """
    删除 <think>...</think> 及其中的所有内容
    """
    if not isinstance(text, str):
        return text

    # 非贪婪匹配，支持跨行
    pattern = re.compile(r'<think>[\s\S]*?</think>', re.IGNORECASE)
    return pattern.sub('', text).strip()


def center_distance_cdf_reward(embeddings: List) -> List[float]:
    """
    embeddings: (N, D)
    return: scores in [0, 1], shape (N,)
    """
    embeddings = np.array(embeddings, dtype=np.float32)
    
    assert embeddings.ndim == 2
    
    if len(embeddings) == 1:
        return [0.0]

    # 1. 均值中心
    center = embeddings.mean(axis=0, keepdims=True)

    # 2. 到中心的欧氏距离
    distances = np.linalg.norm(embeddings - center, axis=1)

    # 3. 经验 CDF
    ranks = rankdata(distances, method="average")  # 1..N
    scores = (ranks - 1) / (len(ranks) - 1)

    return scores.tolist()


def center_distance_sigmoid_scores(embeddings, temperature=1.0):
    emb = np.asarray(embeddings, dtype=np.float32)
    center = emb.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(emb - center, axis=1)

    z = (distances - distances.mean()) / (distances.std() + 1e-8)
    scores = 1 / (1 + np.exp(-z / temperature))

    return scores.tolist()


def center_distance_linear_scores(embeddings) -> List[float]:
    """
    embeddings: (N, D) list or np.ndarray
    return: List[float] of length N, values in [0, 1]
    """
    # 转为 numpy array
    emb = np.asarray(embeddings, dtype=np.float32)

    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")

    N = emb.shape[0]
    if N == 0:
        return []
    if N == 1:
        return [0.0]

    # 1️⃣ 均值中心
    center = emb.mean(axis=0, keepdims=True)

    # 2️⃣ 到中心的欧氏距离
    distances = np.linalg.norm(emb - center, axis=1)

    d_min = distances.min()
    d_max = distances.max()

    # 3️⃣ 线性映射到 [0, 1]
    if d_max - d_min < 1e-8:
        # 所有点距离相同
        scores = np.zeros(N, dtype=np.float32)
    else:
        scores = (distances - d_min) / (d_max - d_min)

    return scores.tolist()


def get_message(role_name, role_desc, user_question, answer, language):
    if language == 'cn':
        prompt = USER_CN.format(
            role_name = role_name, 
            role_desc = role_desc, 
            user_question = user_question, 
            answer = answer
        )
        return [
            {"role": "system", "content": SYSTEM_CN},
            {"role": "user", "content": prompt}
        ]
    elif language == 'en':
        prompt = USER_EN.format(
            role_name = role_name, 
            role_desc = role_desc, 
            user_question = user_question, 
            answer = answer
        )
        return [
            {"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": prompt}
        ]
    else:
        raise ValueError("错误的language输入")
    
    
def extract_score(s):
    match = re.search(r'\[\[([0-5])\]\]', s)
    return int(match.group(1)) if match else None

    
def predict(messages): 
    """调用部署的 vLLM(OpenAI 兼容) 服务生成响应。

    优先从环境变量读取：
      - VLLM_API_BASE: 例如 http://127.0.0.1:8355/v1
      - VLLM_API_KEY: 例如 EMPTY 或你的鉴权KEY
    如未设置，则回落到用户给定的默认值。
    """
    # api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8355/v1") 
    api_base = VLLM_API_BASE
    api_key  = os.environ.get("VLLM_API_KEY", "EMPTY")

    if OpenAI is None:
        raise RuntimeError("openai SDK 未安装，无法调用兼容接口。请先 pip install openai>=1.0.0")

    client = None
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
    except Exception as e:
        raise RuntimeError(f"初始化 OpenAI 客户端失败: {e}")

    max_try = 5
    last_err = None
    while max_try > 0:
        try:
            # 获取服务端可用模型
            model_name = client.models.list().data[0].id

            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.5,
                top_p=0.9,
                max_tokens=2048,
                extra_body={
                    "repetition_penalty": 1.05,
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            
            return extract_score(resp.choices[0].message.content.strip())
        
        except Exception as e:
            last_err = e
            max_try -= 1
            print(f"[ERROR] predict调用失败，重试中... 剩余{max_try}次。错误: {e}")
            
    raise RuntimeError(f"predict多次重试失败: {last_err}")