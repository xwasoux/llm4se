import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

def umap(embeddings: list, idx: list, dim: int = 2) -> list:
    df = pd.DataFrame(data=embeddings, index=idx)
    reducer = UMAP(n_components=dim)
    return reducer.fit_transform(df)

def tsne(embeddings: list, idx: list, dim: int = 2) -> list:
    df = pd.DataFrame(data=embeddings, index=idx)
    reducer = TSNE(n_components=dim)
    return reducer.fit_transform(df)

def pca(embeddings: list, idx: list, dim: int = 2) -> list:
    df = pd.DataFrame(data=embeddings, index=idx)
    reducer = PCA(n_components=dim)
    return reducer.fit_transform(df)
