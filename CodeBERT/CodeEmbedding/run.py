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

def select_pooler(vectors, pooling_type="cls"):
    if pooling_type == "cls":
        return vectors[0]
    elif pooling_type == "mean":
        return torch.mean(input=torch.from_numpy(vectors[1:-1]).clone(), dim=1)
    elif pooling_type == "max":
        return torch.max(input=torch.from_numpy(vectors[1:-1]).clone(), dim=1)

class EmbeddingAnalyser:
    def __init__(self, codes:list, embeddings:list, labels:list) -> None:
        if len(embeddings) != len(labels):
            ValueError()

        self.codes = codes
        self.embeddings = embeddings
        self.labels = labels

    def reduce_dimension(self, method="umap", dim=2, pooling="cls") -> None:
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
        print(self.compressed_df)

    def plot_embedding(self, file_name:str="plotEmbedding.html") -> None:
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

    def conineSimilarity(self, dir:str, fileName:str="cosSim.csv"):
        pairList = []
        cos = torch.nn.CosineSimilarity(dim=0)

        self.svectors = [torch.from_numpy(svector).clone() for svector in self.all_pooler_vectors]

        for label, vector in zip(self.labels, self.svectors):
            for lab, vec in zip(self.labels, self.svectors):
                eachLine = {}
                eachLine["index"] = f"{label}_{lab}"
                eachLine["source"] = label
                eachLine["target"] = lab
                
                eachLine["cosSim"] = float(cos(vector, vec))
                
                pairList.append(eachLine)
        
        df = pd.DataFrame(pairList)
        df.to_csv(f"{dir}/{fileName}")

        return None

    def clustering(self, algo="heirarchal"):
        if algo == "heirarchal":
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

    def plot_clustering(self, file_name="plot_clustering.html") -> None:
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

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    code_list, label_list = sep_csv(args.csv_data_path)

    code_encoder = CodeBertEncoder()
    code_encoder.input(sentences=code_list)

    logging.info("Embedding Codes...")
    embeddings = code_encoder.embedding()

    ## Todo: separate class (named Analyser)
    logging.info(f"Reducing to {args.dimention} Dimension...")
    analyser = EmbeddingAnalyser(codes=code_list,
                        embeddings=embeddings, labels=label_list)
    analyser.reduce_dimension(dim=args.dimention)

    logging.info(f"Plotting Destributed Representation of Codes...")
    analyser.plot_embedding()

    logging.info(f"Calculating Cosine Similarity of Code pairs...")
    analyser.conineSimilarity(dir=args.outputdir)

    return None


if __name__ == "__main__":
    main()