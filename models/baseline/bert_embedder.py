"""
BERT embedding utilities: mean-pooling of last hidden states.
"""
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List

class BertSentenceEmbedder:
    def __init__(self, model_name: str = "bert-base-uncased", device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 16):
        """
        Args:
            texts (List[str]): input sentences
            batch_size (int): number of sentences per batch
        Returns:
            np.ndarray: shape (len(texts), hidden_dim)
        """
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # (B, T, H)
            mask = inputs["attention_mask"].unsqueeze(-1)  # (B, T, 1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            emb = summed / counts
            all_embs.append(emb.detach().cpu())
        import numpy as np
        return np.vstack([e.numpy() for e in all_embs])
