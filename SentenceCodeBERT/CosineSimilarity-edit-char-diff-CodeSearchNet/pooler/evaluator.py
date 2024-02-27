import re
import os
import sys
import csv
import glob
import gzip
import math
import pickle
import random
import codecs
import logging
import argparse
import Levenshtein
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers import models, losses, evaluation

import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def extract_code_idx(args: argparse, index_data: pd.DataFrame, test_data_df: pd.DataFrame, retrieve_size: int = 10) -> list:
    unique_code_idxs = test_data_df['idx1'].unique()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    ## Tokenize code and retrieve code less than 512 tokens.
    less_than_512_code = {}
    for idx in tqdm(unique_code_idxs):
        code = index_data.loc[idx, "func"]
        tokenized_code = tokenizer.tokenize(text=code, max_length=512)
        if len(tokenized_code) < 512:
            less_than_512_code[idx] = code

    ## Sort by number of characters, and retrieve the top X0.
    sorted_less_than_512_code = sorted(less_than_512_code.items(), key=lambda x: len(x[1]), reverse=True)
    candidate_code_idxs = [idx for idx, code in sorted_less_than_512_code[:retrieve_size]]
    return candidate_code_idxs

def encode_and_calculate_cosine_similarity(args: argparse, index_data: pd.DataFrame, test_data_df: pd.DataFrame) -> pd.DataFrame:
    model_paths = [args.mean_pooling_model_path, args.max_pooling_model_path, args.cls_pooling_model_path]
    model_labels = ["mean", "max", "cls"]

    for model_path, model_label in zip(model_paths, model_labels):
        ## Load file-tuned model
        model = SentenceTransformer(model_path)

        ## Encode and calculate cosine similarity
        for _, row in test_data_df.iterrows():
            idx1 = row['idx1']
            idx2 = row['idx2']
            text1 = index_data.loc[idx1, "func"]
            text2 = index_data.loc[idx2, "func"]

            text1_embedding = model.encode(text1)
            text2_embedding = model.encode(text2)

            cosine_score = util.cos_sim(text1_embedding, text2_embedding).cpu().numpy().item()
            test_data_df.loc[_, f"{model_label}_cosine_score"] = cosine_score
            test_data_df.loc[_, "edit_code"] = text2
            test_data_df.loc[_, "original_code"] = text1
    return test_data_df

def correlation_coefficient(x: pd.DataFrame, y: pd.DataFrame) -> float:
    assert len(x) == len(y)
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = math.sqrt(((x - x_mean)**2).sum() * ((y - y_mean)**2).sum())
    return numerator / denominator

def plot_all_line_diagram(df: pd.DataFrame, pruning_type: str, save_dir: str) -> None:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['idx2'], y=df['edit_distance'], mode='lines', name='Edit Distance', line=dict(color="#5656d6")), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['idx2'], y=df['mean_cosine_score'], mode='lines', name='MEAN Pooling', line=dict(color="#1db540")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df['idx2'], y=df['max_cosine_score'], mode='lines', name='MAX Pooling', line=dict(color="#b5a11d")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df['idx2'], y=df['cls_cosine_score'], mode='lines', name='CLS Pooling', line=dict(color="#e02430")), secondary_y=True)
    fig.update_layout(width=1200, height=1200, title_text=f"Edit Distance and Cosine Similarity Score ({pruning_type})")

    pooler = "all"
    fig.write_html(os.path.join(save_dir, f"line-plot_{pruning_type}-{pooler}.html"))
    fig.write_image(os.path.join(save_dir, f"line-plot_{pruning_type}-{pooler}.png"))
    return None

def plot_each_line_diagram(df: pd.DataFrame, pruning_type: str, pooler: str, save_dir: str) -> None:
    if pooler == "mean":
        color = "#1db540"
    elif pooler == "max":
        color = "#b5a11d"
    elif pooler == "cls":
        color = "#e02430"
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['idx2'], y=df['edit_distance'], mode='lines', name='Edit Distance', line=dict(color="#5656d6")), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['idx2'], y=df[f"{pooler}_cosine_score"], mode='lines', name=f"{pooler} Cosine Score", line=dict(color=color)), secondary_y=True)
    fig.update_layout(width=1200, height=1200, title_text=f"Edit Distance and Cosine Similarity Score ({pruning_type})")
    
    fig.write_html(os.path.join(save_dir, f"line-plot_{pruning_type}-{pooler}.html"))
    fig.write_image(os.path.join(save_dir, f"line-plot_{pruning_type}-{pooler}.png"))
    return None

def judge_compilable_code(args: argparse, df: pd.DataFrame):
    if args.language == "python":
        import ast
    
    for _, row in df.iterrows():
        code = row['edit_code']
        try:
            if args.language == "python":
                ast.parse(code)
            df.loc[_, "compilable"] = True
        except:
            df.loc[_, "compilable"] = False
    return df

def plot_histogram(df: pd.DataFrame, pruning_type: str, save_dir: str, cosine_score: str = "mean_cosine_score") -> None:
    fig = ff.create_distplot([df[df['compilable'] == True][cosine_score], df[df['compilable'] == False][cosine_score]],
                            group_labels=["Compilable", "Uncompilable"],
                            bin_size=0.01,
                            curve_type="normal",
                            colors=["#1db540", "#e02430"])
    fig.write_html(os.path.join(save_dir, f"histgram_{pruning_type}_mean.html"))
    fig.write_image(os.path.join(save_dir, f"histgram_{pruning_type}_mean.png"))
    return None

def calculate_basic_statistics(df: pd.DataFrame, pooler: str) -> list:
    cosine_score = f"{pooler}_cosine_score"
    count = df[cosine_score].count()
    mean = df[cosine_score].mean()
    variance = df[cosine_score].var()
    std = df[cosine_score].std()
    return [count, mean, variance, std]

def main() -> None:
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--mean_pooling_model_path', type=str, required=True)
    parser.add_argument('--max_pooling_model_path', type=str, required=True)
    parser.add_argument('--cls_pooling_model_path', type=str, required=True)
    parser.add_argument("--index_data_file", type=str, default=None, required=True)
    parser.add_argument('--test_data_file', type=str, required=True)

    ## Other parameters
    parser.add_argument('--retrieve_size', type=int, default=10)
    parser.add_argument('--language', type=str, default="python")
    parser.add_argument('--output_dir', type=str, default="./output")

    args = parser.parse_args()

    ## Validation each file exists
    assert os.path.exists(args.test_data_file)
    assert os.path.exists(args.mean_pooling_model_path)
    assert os.path.exists(args.max_pooling_model_path)
    assert os.path.exists(args.cls_pooling_model_path)

    ## Create logger
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    index_data = pd.read_json(args.index_data_file, lines=True, orient="records", encoding="utf-8").set_index("idx")
    test_data_df = pd.read_csv(args.test_data_file, sep='\t')
    candidate_code_idxs = extract_code_idx(args=args, index_data=index_data, test_data_df=test_data_df, retrieve_size=args.retrieve_size)
    
    for rank, idx in enumerate(candidate_code_idxs):
        ## Extract target code from test data dataframe and separate dataframe
        target_codes_df = test_data_df[test_data_df['idx1'] == idx]

        ## Create save directory
        save_directory = os.path.join(args.output_dir, "%s=%s" % (rank, idx))
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        ## Encode and calculate cosine similarity
        filename=os.path.join(save_directory, "log.txt")
        sys.stdout = open(filename, 'w')
        print("********** Running test **********")
        print("\tNum examples =", len(target_codes_df))
        print("\tMean pooling model:", args.mean_pooling_model_path)
        print("\tMax pooling model:", args.max_pooling_model_path)
        print("\tCLS pooling model:", args.cls_pooling_model_path)
        print("\tEncode function name:", idx)
        original_code = index_data.loc[idx, "func"]
        print("\tString length:", len(original_code))
        print("\tEncode function: \n", original_code)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tokenized_code = tokenizer.tokenize(text=original_code, max_length=512)
        print("\tTokenized string length:", len(tokenized_code))
        print("\tTokenized string: \n", tokenized_code)
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    
        target_codes_df = encode_and_calculate_cosine_similarity(args=args, index_data=index_data, test_data_df=target_codes_df)
        sequence_df = target_codes_df[target_codes_df['pruning_type'] == "sequence"]
        subtree_df = target_codes_df[target_codes_df['pruning_type'] == "subtree"]

        # Create plot figure
        # sequence data
        plot_all_line_diagram(sequence_df, pruning_type="sequence", save_dir=save_directory)
        # subtree data
        plot_all_line_diagram(subtree_df, pruning_type="subtree", save_dir=save_directory)

        ## Plot line diagram from each pooler model
        model_labels = ["mean", "max", "cls"]
        for pooler in model_labels:
            ## sequence data
            plot_each_line_diagram(sequence_df, pruning_type="sequence", pooler=pooler, save_dir=save_directory)
            ## subtree data
            plot_each_line_diagram(subtree_df, pruning_type="subtree", pooler=pooler, save_dir=save_directory)

        ## Calculate Pearson correlation coefficient from each pooler model and save result
        all_pooler_cc_table = []
        for df, partition in zip((sequence_df, subtree_df), ("Sequencial Pruning Data", "Subtree Pruning Data")):
            mean_pearson_score = correlation_coefficient(df['edit_distance'], df['mean_cosine_score'])
            max_pearson_score = correlation_coefficient(df['edit_distance'], df['max_cosine_score'])
            cls_pearson_score = correlation_coefficient(df['edit_distance'], df['cls_cosine_score'])
            all_pooler_cc_table.append([partition, mean_pearson_score, max_pearson_score, cls_pearson_score])
        res_df = pd.DataFrame(all_pooler_cc_table, columns=["partition", "mean_pearson_score", "max_pearson_score", "cls_pearson_score"])
        res_df.to_csv("{}.tsv".format(os.path.join(save_directory, "pearson-score-table")), sep='\t', index=False, header=True)

        ## Judge compilable code or not and save result
        target_codes_df = judge_compilable_code(args=args, df=target_codes_df)
        sequence_df = target_codes_df[target_codes_df['pruning_type'] == "sequence"]
        subtree_df = target_codes_df[target_codes_df['pruning_type'] == "subtree"]
        plot_histogram(sequence_df, pruning_type="sequence", save_dir=save_directory)
        plot_histogram(subtree_df, pruning_type="subtree", save_dir=save_directory)

        ## Calculate basic statistics and save result
        all_pooler_histgram_table = []
        for pooler in model_labels:
            for df, partition in zip((sequence_df, subtree_df), ("Sequencial Pruning Data", "Subtree Pruning Data")):
                compilable_df = df[df['compilable'] == True]
                un_compilable_df = df[df['compilable'] == False]
                compilable_stat_list = calculate_basic_statistics(df=compilable_df, pooler=pooler) ## use mean_cosine_score only
                un_compilable_stat_list = calculate_basic_statistics(df=un_compilable_df, pooler=pooler) ## use mean_cosine_score only
                each_list = ["%s - %s" % (partition, pooler)] + compilable_stat_list + un_compilable_stat_list
                all_pooler_histgram_table.append(each_list)
        histgram_df = pd.DataFrame(all_pooler_histgram_table, columns=["partition", 
                                                                        "compilable_count", "compilable_mean", "compilable_variance", "compilable_std", 
                                                                        "un_compilable_count", "un_compilable_mean", "un_compilable_variance", "un_compilable_std"])
        histgram_df.to_csv("{}.tsv".format(os.path.join(save_directory, "histgram-table")), sep='\t', index=False, header=True)

        ## Save result
        target_codes_df.to_csv("{}.tsv".format(os.path.join(save_directory, "result")), sep='\t', index=False, header=True)
    return None

if __name__ == '__main__':
    main()
