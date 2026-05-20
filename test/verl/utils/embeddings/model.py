import requests

class EmbeddingVLLM():
    def __init__(self, url):
        self.url = url
        
    def encode(self, data, batch_size=64, show_progress_bar=False, normalize_embeddings=False):
        record = {
        "texts": data,
        "batch_size": batch_size,
        "show_progress_bar": show_progress_bar,
        "normalize_embeddings": normalize_embeddings
        }
        
        res = requests.post(self.url, json=record)
        
        return res.json()["embeddings"]