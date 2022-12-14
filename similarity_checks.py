from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
class SimilarityChecks:
    def __init__(
        self, transformer_model_base="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    ) -> None:
        self.base_model = SentenceTransformer(transformer_model_base)

    def compute(self, source, targets):
        trg_embedding = self.encode(targets)
        src_embedding = self.encode(source)
        scores = util.pytorch_cos_sim(src_embedding, trg_embedding).cpu().numpy()
        return scores

    def encode(self, x):
        return self.base_model.encode(x)

    def top_k_targets(self, source, targets, k=2):
        scores = self.compute(source, targets)[0]
        trg_scores = [(q, s) for (q, s) in zip(targets, scores)]
        trg_scores = sorted(trg_scores, key=lambda x: x[-1], reverse=True)[:k]
        return [t[0] for t in trg_scores]

    def semantic_search(self, source, documents, top_k=10):
        docs_embedding = self.encode(documents)
        src_embedding = self.encode(source)
        searches = util.semantic_search(
            src_embedding, docs_embedding, score_function=util.cos_sim, top_k=top_k
        )[0]
        return searches