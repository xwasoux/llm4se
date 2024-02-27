import re
import os
import sys
import csv
import gzip
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Iterable, List, Tuple, Dict, Union, Optional, Any, NamedTuple, Callable, Iterator, TypeVar

import torch
from sentence_transformers import SentenceTransformer

import git
from git import Repo
from tree_sitter import Language, Parser
from data.parser import remove_comments_and_docstrings, get_functions_or_methods
from utils import umap, tsne, pca

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


file_extension = {
    "go": [".go"],
    "javascript": [".js"],
    "python": [".py"],
    "php": [".php"],
    "java": [".java"],
    "ruby": [".rb"],
    "c-sharp": [".cs"],
    "c": [".c"],
    "c++": [".cpp", ".cxx", ".cc"],
}

def get_user_and_repo_name(args: argparse) -> str:
    res = re.findall(r'(https://github.com/|git@github.com:)([a-zA-Z0-9\-]*)/([a-zA-Z0-9\-]*)(.git)?', args.remote_repo_url)
    user = res[0][1]
    repo = res[0][2]
    return user, repo

def clone_from_github(args: argparse.Namespace) -> git.repo.base.Repo:
    username, reponame = get_user_and_repo_name(args=args)
    clone_path = os.path.join(os.path.dirname(__file__), "data", "clone", username, reponame)
    args.local_repo_path = clone_path
    if not os.path.exists(clone_path):
        os.makedirs(clone_path, exist_ok=True)
        return Repo.clone_from(url=args.remote_repo_url, to_path=clone_path)
    else:
        return Repo(clone_path)

def retreve_target_files(args: argparse.Namespace, working_dir: str) -> list:
    return glob.glob(os.path.join(working_dir, "**", f"*.{args.language}"), recursive=True)

def remove_abs_path(file_path: str, user_and_repo_name:str) -> str:
    ancectors = re.match(rf'.*({user_and_repo_name})/', file_path)
    return file_path.replace(ancectors.group(0), "")

def preprocess_from_repo(args: argparse.Namespace, repo: Repo, user_and_repo_name: str, input_examples_file: str) -> pd.DataFrame:
    ## Get all files in repository
    all_target_files = retreve_target_files(args=args, working_dir=repo.working_dir)

    ## Parser
    LANGUAGE = Language(os.path.join("data", "parser", "my-languages.so"), args.language)
    parser = Parser()
    parser.set_language(LANGUAGE)

    ## Preprocess codes
    logger.info("***** Preprocess codes *****")
    logger.info(f"\tRepository: {user_and_repo_name}")
    logger.info(f"\tNumber of files: {len(all_target_files)}")
    logger.info(f"\tLanguage: {args.language}")
    logger.info(f"\tStart preprocessing...")
    func_list = []
    idx = 0
    for target_file in tqdm(all_target_files):
        with open(target_file) as f:
            code = f.read()
        tree = parser.parse(bytes(code, "utf8"))
        functions = get_functions_or_methods(tree=tree, function=True, method=True)
        for func in functions:
            func_info = {}
            func_info["idx"] = idx
            func_info["repo"] = user_and_repo_name
            func_info["path"] = remove_abs_path(target_file, user_and_repo_name)
            func_info["language"] = args.language
            func_info["original_string"] = func.text.decode()
            func_info["cleaned_code"] = remove_comments_and_docstrings(func.text.decode(), args.language)
            func_list.append(func_info)
            idx += 1

    df = pd.DataFrame(func_list)
    df.to_json(input_examples_file, force_ascii=False, lines=True, orient='records')
    return df

def load_and_cache_examples(args: argparse.Namespace) -> pd.DataFrame:
    if args.local_repo_path:
        repo = Repo(args.local_repo_path)
    if args.remote_repo_url:
        repo = clone_from_github(args)

    ## Get user and repo name
    username, reponame = get_user_and_repo_name(args=args)
    user_and_repo_name = f"{username}/{reponame}"

    ## Load or create inputs with preprocessing...
    input_examples_file = os.path.join(args.output_dir, f"{user_and_repo_name}", "inputs.jsonl")
    os.makedirs(os.path.dirname(input_examples_file), exist_ok=True)
    if not os.path.exists(input_examples_file):
        df = preprocess_from_repo(args=args, repo=repo, user_and_repo_name=user_and_repo_name, input_examples_file=input_examples_file)
    else:
        df = pd.read_json(input_examples_file, orient='records', lines=True)

    ## Set output_dir
    if len(args.model_name_or_path.split("/")) == 2:
        args.output_dir = os.path.join(args.output_dir, user_and_repo_name, args.model_name_or_path.replace("/", "-"))
    elif len(args.model_name_or_path.split("/")) > 2:
        args.output_dir = os.path.join(args.output_dir, user_and_repo_name, args.model_name_or_path.split("/")[-1])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"****** Load functions from Repository ******")
    logger.info(f"\tRepository: {user_and_repo_name}")
    logger.info(f"\tNumber of functions: {len(df)}")
    logger.info(f"\tNumber of files: {len(df['path'].unique())}")

    return df

class MyDataset:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.idx = dataset["idx"].tolist()
        self.path = dataset["path"].tolist()
        self.sentences = dataset["cleaned_code"].tolist()
        self.embeddings = []
        self.umap_embeddings = []
        self.t_sne_embeddings = []
        self.pca_embeddings = []
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx], self.embeddings[idx]


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

def hierarchal_clustering(args: argparse.Namespace, embeddings: list, img_file_name: str = "hierarchal_dendrogram.png",
                                    plot_html: bool = True, width: int = 800, height: int = 500) -> None:
    logger.info("***** Start hierarchal clustering *****")
    hierarchal = ff.create_dendrogram(embeddings)
    hierarchal.update_layout(width=width, height=height)

    img_file_name = os.path.join(args.output_dir, img_file_name)
    hierarchal.write_image(img_file_name)
    if plot_html:
        hierarchal.write_html(img_file_name.replace(".png", ".html"))
    return None

def _create_cluster_labels(cluster_num_labels: list) -> list:
    cluster_labels = []
    for cluster in cluster_num_labels:
        if cluster == -1: 
            label = "noise"
        else:
            label = f'cluster-{cluster}'
        cluster_labels.append(label)
    return cluster_labels

def hdbscan_clustering(args: argparse.Namespace, embeddings: list, idx: list, sentences: list, dataset: MyDataset, 
                        img_file_name: str = "hdbscan_scatter.png", plot_html: bool = True, width: int = 800, height: int = 500) -> None:
    logger.info("***** Start hdbscan clustering *****")
    hdbscan = HDBSCAN()
    hdbscan.fit(embeddings)

    ## Create cluster labels for plotting
    cluster_labels = _create_cluster_labels(hdbscan.labels_)
    logging.info(f"\tNumber of clusters: {len(set(cluster_labels))}")

    df = pd.DataFrame(data=embeddings, index=idx)
    df["cluster"] = cluster_labels
    df["sentences"] = sentences
    df["path"] = dataset.path
    scatter_fig = px.scatter(data_frame=df, x=0, y=1, color=cluster_labels, symbol=cluster_labels, 
                                                            hover_data=["cluster", "sentences", "path"])
    scatter_fig.update_layout(width=800, height=500)
    scatter_fig.update_layout(hoverlabel_align = "left")

    img_file_name = os.path.join(args.output_dir, img_file_name)
    scatter_fig.write_image(img_file_name)
    if plot_html:
        scatter_fig.write_html(img_file_name.replace(".png", ".html"))
    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
    parser.add_argument("--output_dir", type=str, default='./output')

    parser.add_argument("--local_repo_path", type=str, default=None)
    parser.add_argument("--remote_repo_url", type=str, default=None)
    parser.add_argument("--language", type=str, default="java")
    parser.add_argument("--dim", type=int, default=2)

    args = parser.parse_args()


    ## Preprocess and load inputs
    inputs_df = load_and_cache_examples(args)
    my_dataset = MyDataset(dataset=inputs_df)

    ## Load model and encode
    model = SentenceTransformer(args.model_name_or_path, device='cuda')
    model.max_seq_length = 512
    my_dataset.embeddings = model.encode(my_dataset.sentences)

    ## Clustering
    logger.info("***** Start clustering *****")
    my_dataset.umap_embeddings = umap(embeddings=my_dataset.embeddings, idx=my_dataset.idx, dim=args.dim)
    hdbscan_clustering(args=args, embeddings=my_dataset.umap_embeddings, idx=my_dataset.idx, sentences=my_dataset.sentences, dataset=my_dataset)
    hierarchal_clustering(args=args, embeddings=my_dataset.umap_embeddings)

    return None


if __name__ == "__main__":
    main()