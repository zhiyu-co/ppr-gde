import pickle
import logging
import os

import torch
import hashlib
from openai import OpenAI

from verl.utils.reward_score.roleplay import predict


SYS_PROMPT = """你是一个语言学专家，你需要使用你所有的语言知识完成用户给予的任务！"""

EN_SYS = """We are evaluating answers to the question"{question}"
Here are two possible answers:
Possible Answer 1: {text1}
Possible Answer 2: {text2}
Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."""

CN_SYS = """我们正在评估问题"{question}"的答案
这里有两个可能的答案：
答案 1: {text1}
答案 2: {text2}
答案 1 是否在语义上蕴含答案 2？请回答 *蕴含* 、 *矛盾* 或 *中性* 。
"""

##### 调用 vllm 获得得分
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)


class BaseEntailment:
    def save_prediction_cache(self):
        pass
    
    
class EntailmentLLM(BaseEntailment):

    entailment_file = 'entailment_cache.pkl'

    def __init__(self, entailment_cache_path=None, entailment_cache_only=False):
        """
        entailment_cache_path:
            - None → 不加载缓存
            - 路径目录 → 将在该目录中寻找 / 保存 entailment_cache.pkl
        """

        self.cache_dir = entailment_cache_path
        self.entailment_cache_only = entailment_cache_only
        self.cache_path = None

        if entailment_cache_path is not None:
            os.makedirs(entailment_cache_path, exist_ok=True)  # 确保目录存在
            self.cache_path = os.path.join(entailment_cache_path, self.entailment_file)

        self.prediction_cache = self.init_prediction_cache(self.cache_path)

    def init_prediction_cache(self, cache_path):
        """从缓存文件中初始化缓存内容"""
        if cache_path is None or not os.path.exists(cache_path):
            print("="*20 + "No cache found. Starting with empty cache." + "="*20)
            return {}

        try:
            with open(cache_path, "rb") as infile:
                cache = pickle.load(infile)
                print("✅ " + f"Read prediction cache from {cache_path}") 
                return cache
        except Exception as e:
            print("="*20 + f"Failed to load cache: {e}" + "="*20)
            return {}

    def save_prediction_cache(self):
        """保存缓存至本地文件"""
        if self.cache_path is None:
            print("="*20 + "No cache path specified. Skip saving cache." + "="*20)
            return

        try:
            with open(self.cache_path, "wb") as outfile:
                pickle.dump(self.prediction_cache, outfile)
            print("✅ " + f"Saved prediction cache to {self.cache_path}") 
        except Exception as e:
            print("="*20 + f"Failed to save cache: {e}" + "="*20)

    def check_implication(self, text1, text2, example=None):
        """ 返回评估分数，若已缓存则直接返回，否则调用模型推理 """
        if example is None:
            raise ValueError

        prompt = self.equivalence_prompt(text1, text2, example)

        hashed = md5hash(prompt)

        if hashed in self.prediction_cache:
            return self.prediction_cache[hashed]

        if self.entailment_cache_only:
            raise ValueError("Cache-only mode but prediction not found in cache.")

        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt}
        ]
        response = self.predict(messages, temperature=0.02)

        binary_response = response.lower()[:30]
        if '蕴含' in binary_response:
            score = 2
        elif '中性' in binary_response:
            score = 1
        elif '矛盾' in binary_response:
            score = 0
        else:
            logging.warning("MANUAL NEUTRAL!")
            score = 1

        self.prediction_cache[hashed] = score

        return score
        

class EntailmentVLLM(EntailmentLLM):

    def __init__(self, entailment_cache_path, url, entailment_cache_only=False):
        super().__init__(entailment_cache_path, entailment_cache_only)
        self.url = url

    def equivalence_prompt(self, text1, text2, question):
        return CN_SYS.format(question=question, text1=text1, text2=text2)

    def predict(self, messages, temperature=0.5):
        """调用部署的 vLLM(OpenAI 兼容) 服务生成响应。

        优先从环境变量读取：
        - VLLM_API_BASE: 例如 http://127.0.0.1:8355/v1
        - VLLM_API_KEY: 例如 EMPTY 或你的鉴权KEY
        如未设置，则回落到用户给定的默认值。
        """
        api_base = self.url
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
                    temperature=temperature,
                    top_p=0.9,
                    max_tokens=2048,
                    extra_body={
                        "repetition_penalty": 1.05,
                        "skip_special_tokens": False,
                        "spaces_between_special_tokens": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                max_try -= 1
                print(f"[ERROR] predict调用失败，重试中... 剩余{max_try}次。错误: {e}")
        raise RuntimeError(f"predict多次重试失败: {last_err}")