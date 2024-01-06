import sys
import os
import gzip
import csv
import logging
from datetime import datetime
import argparse

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
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from model import CodeBertEncoder

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO
                    )

def sep_csv(csv_data_path):
    code_list = []
    label_list = []

    with open(csv_data_path) as f:
        reader = csv.reader(f)
        reader_list = [row for row in reader]

    for row in reader_list:
        code_list.append(row[0])
        label_list.append(row[1])
    
    return code_list, label_list

class MyDataset:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.label = dataset["repo"].tolist()
        self.data = dataset["cleaned_code"].tolist()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

def select_pooler(vectors, pooling_type: str = "cls"):
    if pooling_type == "cls":
        return vectors[0]
    elif pooling_type == "mean":
        return torch.mean(input=torch.from_numpy(vectors[1:-1]).clone(), dim=1)
    elif pooling_type == "max":
        return torch.max(input=torch.from_numpy(vectors[1:-1]).clone(), dim=1)

class DimReducer:
    def __init__(self, codes: list, embeddings: list, labels: list) -> None:
        if len(embeddings) != len(labels):
            ValueError()

        self.codes = codes
        self.embeddings = embeddings
        self.labels = labels

    def reduce_dimension(self, method: str = "umap", dim: int = 2, pooling: str = "cls") -> pd.DataFrame:
        if not self.embeddings:
            ValueError()
        
        self.dim = dim
        
        self.all_pooler_vectors = []
        for context in self.embeddings:
            last_hs_vector = context["last_hidden_state"][0].to('cpu').detach().numpy().copy()
            self.all_pooler_vectors.append(select_pooler(last_hs_vector, pooling_type=pooling))

        all_code_series_list = []
        for pooler_vector, label in zip(self.all_pooler_vectors, self.labels):
            row = pd.Series(pooler_vector, name=label)
            all_code_series_list.append(row)
            
        concat_df = pd.concat(all_code_series_list, axis=1)
        concat_df = concat_df.T
        df_index = concat_df.index

        if method == "umap":
            dim_reducer = UMAP(n_components=self.dim)
        elif method == "t-sne":
            dim_reducer = TSNE(n_components=self.dim)
        elif method == "pca":
            dim_reducer = PCA(n_components=self.dim)

        self.compressed_vectors = dim_reducer.fit_transform(concat_df)
        self.compressed_df = pd.DataFrame(data=self.compressed_vectors, index=df_index)
        self.compressed_df["Code"] = self.codes
        self.compressed_df["label"] = self.labels
    
        return self.compressed_df

    def plot_embedding(self, file_name: str = "plotEmbedding.html") -> None:
        if self.dim == 2:
            self.embedding_fig = px.scatter(self.compressed_df, 
                                            x=0, 
                                            y=1, 
                                            text=self.compressed_df.index, 
                                            color=self.labels, 
                                            hover_name="label", 
                                            hover_data=["Code"])
        elif self.dim == 3:
            self.embedding_fig = px.scatter_3d(self.compressed_df, 
                                            x=0, 
                                            y=1, 
                                            z=2,  
                                            text=self.compressed_df.index, 
                                            color=self.labels, 
                                            hover_name="label", 
                                            hover_data=["Code"])

        self.embedding_fig.write_html(file_name)


class ClusterAnalyser:
    def __init__(self, compressed_vectors: np.ndarray) -> None:
        self.compressed_vectors = compressed_vectors

    def clustering(self, algo: str = "hierarchal"):
        if algo == "hierarchal":
            pass
        elif algo == "hdbscan":
            self.hdbscan = HDBSCAN().fit(self.compressed_vectors)

            self.hdbscan_labels = self.hdbscan.labels_

            ## make cluster label to List
            self.type_labels = []
            for cls in self.hdbscan_labels:
                if cls == -1: 
                    label = "noise"
                else:
                    label = f'cluster-{cls}'
                self.type_labels.append(label)
            
            self.compressed_df["label_type"] = self.type_labels

    def plot_clustering(self, file_name: str = "plot_clustering.html") -> None:
        if self.dim == 2:
            self.clustering_fig = px.scatter(self.compressed_df, 
                                            x=0, 
                                            y=1, 
                                            text=self.compressed_df.index, 
                                            color="label_type", 
                                            hover_name="label", 
                                            hover_data=["Code"])
        elif self.dim == 3:
            self.clustering_fig = px.scatter_3d(self.compressed_df, 
                                            x=0, 
                                            y=1, 
                                            z=2,  
                                            text=self.compressed_df.index, 
                                            color="label_type", 
                                            hover_name="label", 
                                            hover_data=["Code"])

        self.clustering_fig.write_html(file_name)

    def plot_dendrogram(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    df = pd.read_json(args.data_path, orient='records', lines=True)
    logging.info(df.head())
    my_data = MyDataset(dataset=df)

    encoder = CodeBertEncoder()
    encoder.input(sentences=my_data.data)

    logging.info("Embedding Codes...")
    embeddings = encoder.embedding()

    logging.info(f"Reducing to {args.dim} Dimension...")
    reducer = DimReducer(codes=my_data.data, embeddings=embeddings, labels=my_data.label)
    reducer.reduce_dimension(dim=args.dim)

    logging.info(f"Plotting Destributed Representation of Codes...")
    reducer.plot_embedding()

    return None


if __name__ == "__main__":
    main()