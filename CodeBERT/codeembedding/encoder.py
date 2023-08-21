import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from hdbscan import HDBSCAN

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class CodeBertEncoder:
    base_tokenizer = "microsoft/codebert-base"
    base_model = "microsoft/codebert-base"

    def __init__(self, tokenizer=base_tokenizer, model=base_model) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModel.from_pretrained(model)
        
    def input(self, sentences:list) -> None:
        self.sentences = sentences
        self.tokenized_code_ids = []
        for sentence in sentences:
            tokenize_codes = self.tokenizer.tokenize(sentence)
            tokens_ids = [self.tokenizer.cls_token] + tokenize_codes + [self.tokenizer.sep_token]
            self.tokenized_code_ids.append(self.tokenizer.convert_tokens_to_ids(tokens_ids))

    def embedding(self) -> list:
        self.embeddings = []
        for code_ids in self.tokenized_code_ids:
            self.embeddings.append(self.model(torch.tensor(code_ids)[None, :]))

        return self.embeddings
