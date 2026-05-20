import torch
from typing import List
from verl.utils.char_rm.BaichuanCharRM.modeling_baichuan import BaichuanCharRM
from verl.utils.char_rm.BaichuanCharRM.tokenization_baichuan import BaichuanTokenizer

class RewardModelService:
    def __init__(self, model_path: str, device="cuda", max_seq_length=4096):
        self.max_seq_length = max_seq_length
        self.tokenizer = BaichuanTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" 
        self.base_model = BaichuanCharRM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
    
    def score(self, text: str) -> float:
        """Batch scoring with GPU batching."""
        ids = self.tokenizer.encode(text=text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        if len(ids) > self.max_seq_length:
            ids = ids[-self.max_seq_length:]
        input_ids = torch.tensor(ids).unsqueeze(0).cuda()
        with torch.no_grad():
            score = self.base_model(input_ids=input_ids)[1].item() * 4 + 1
        
        return score
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """Batch scoring with GPU batching."""
        scores = []

        for text in texts:
            ids = self.tokenizer.encode(text=text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            if len(ids) > self.max_seq_length:
                ids = ids[-self.max_seq_length:]
            input_ids = torch.tensor(ids).unsqueeze(0).cuda()
            with torch.no_grad():
                score = self.base_model(input_ids=input_ids)[1].item() * 4 + 1
                scores.append(score)
        
        return scores
