from verl import DataProto
import json
import requests
import os

from verl.workers.reward_manager.tools import is_valid_think_format, remove_think_block


class CharacterRewardManager:
    def __init__(self, url):
        self.url = url
        
    def __call__(self, data: DataProto, step = 0, has_thinking = False):
        """调用远程服务，计算得到 CharacterEval 评估分数"""
        scores = {}
        
        log_dir = os.getenv("LOG_DIR", ".")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"char_results_{step}.jsonl")
        
        for i in range(len(data)):
            data_item = data[i]
            
            output = data_item.non_tensor_batch['output_text']
            extra_info = data_item.non_tensor_batch['extra_info']
            
            if has_thinking:
                if is_valid_think_format(output):
                    output = remove_think_block(output)
                else:
                    record = {
                        "format": "False",
                        "role_info": extra_info['role_info'],
                        "context": extra_info['context'],
                        "model_output": output,
                        "metric_zh": extra_info['metric_zh']
                    }
                    scores.setdefault(extra_info['metric_en'], []).append(0.0)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue
                    
            output = output.split("\n")[0]
            
            record = {
                "role_info": extra_info['role_info'],
                "context": extra_info['context'],
                "model_output": output,
                "metric_zh": extra_info['metric_zh']
            }
            
            res = requests.post(self.url, json=record)
            
            scores.setdefault(extra_info['metric_en'], []).append(res.json()['score'])
            
            record["score"] = res.json()['score']
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        return scores
        
        