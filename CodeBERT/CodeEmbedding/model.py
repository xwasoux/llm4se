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

from tqdm import tqdm

class CodeBertEncoder:
    base_tokenizer = "microsoft/codebert-base"
    base_model = "microsoft/codebert-base"

    def __init__(self, tokenizer=base_tokenizer, model=base_model) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModel.from_pretrained(model)
        self.max_length = 512

    def embedding(self, sentences: list) -> list:
        self.sentences = sentences
        self.token_ids_list = []
        self.embeddings = []

        for sentence in tqdm(self.sentences):
            tokenized_codes = self.tokenizer.tokenize(sentence)
            tokens = [self.tokenizer.cls_token] + tokenized_codes + [self.tokenizer.sep_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.token_ids_list.append(token_ids)

            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            self.embeddings.append(self.model(torch.tensor(token_ids)[None, :]))

        return self.embeddings
