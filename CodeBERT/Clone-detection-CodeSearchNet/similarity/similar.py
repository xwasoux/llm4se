import re
import os
import sys
import csv
import glob
import gzip
import math
import pickle
import random
import logging
import argparse
import Levenshtein
import numpy as np
import pandas as pd
import code_diff as cd
import code_tokenize as ctok
from datetime import datetime
from tqdm import tqdm, trange
from multiprocessing import Pool 

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_similarity_table_parallel(args):
    df, row = args
    idx1 = row['idx1']
    idx2 = row['idx2']
    idx = f"{idx1}-{idx2}"

    text1 = row['text1']
    text2 = row['text2']
    clone_label = row['clone_label']
    prediction = row['prediction']

    true_positive = 1 if (clone_label == 1) and (prediction == 1) else 0
    false_positive = 1 if (clone_label == 0) and (prediction == 1) else 0
    false_negative = 1 if (clone_label == 1) and (prediction == 0) else 0
    true_negative = 1 if (clone_label == 0) and (prediction == 0) else 0

    char_diff = Levenshtein.distance(text1, text2)
    char_diff_ratio = 1 - (char_diff / max(len(text1), len(text2)))

    text1_tokens = ctok.tokenize(text1, lang="java", syntax_error="ignore")
    text1_tokens = [str(t) for t in text1_tokens]
    text2_tokens = ctok.tokenize(text2, lang="java", syntax_error="ignore")
    text2_tokens = [str(t) for t in text2_tokens]
    token_diff = Levenshtein.distance(text1_tokens, text2_tokens)
    token_diff_ratio = 1 - (token_diff / max(len(text1_tokens), len(text2_tokens)))

    edit_script = cd.difference(text1, text2, lang="java").edit_script()
    ast_diff = len(edit_script)
    return [idx, text1, text2, clone_label, prediction, true_positive, false_positive, false_negative, true_negative, 
            char_diff, char_diff_ratio, token_diff, token_diff_ratio, str(edit_script), ast_diff]

def create_similarity_table(df: pd.DataFrame) -> pd.DataFrame:
    additional_table = []
    args_list = [(df, row) for _, row in df.iterrows()]
    
    with Pool() as pool:
        additional_table = list(tqdm(pool.imap(create_similarity_table_parallel, args_list), total=len(args_list)))

    similarity_df = pd.DataFrame(additional_table, columns=["idx", "text1", "text2", "clone_label", "prediction", "true_positive", "false_positive", "false_negative", "true_negative",
                                                            "char_diff", "char_diff_ratio", "token_diff", "token_diff_ratio", "edit_script", "ast_diff"])
    return similarity_df

def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats_table = []
    for name, key in zip(["TP", "FP", "FN", "TN"], ["true_positive", "false_positive", "false_negative", "true_negative"]):
        for col in ["char_diff", "char_diff_ratio", "token_diff", "token_diff_ratio", "ast_diff"]:
            stats_table.append([f"{name}_{col}", df[df[key]==1][col].count(), df[df[key]==1][col].mean(), df[df[key]==1][col].median(), 
                                df[df[key]==1][col].min(), df[df[key]==1][col].max(), df[df[key]==1][col].var(), df[df[key]==1][col].std()])
    stats_df = pd.DataFrame(stats_table, columns=["name", "count", "mean", "median", "min", "max", "var", "std"])
    return stats_df

def plot_all_line_diagram(df: pd.DataFrame, basename: str, name: str) -> None:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["idx"], y=df['char_diff'],         mode='lines', name='Character Edit Distance', line=dict(color="#5656d6")), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["idx"], y=df['char_diff_ratio'],   mode='lines', name='Ratio of Character Edit Distance', line=dict(color="#1db540")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["idx"], y=df['token_diff'],        mode='lines', name='Token Edit Distance', line=dict(color="#b5a11d")), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["idx"], y=df['token_diff_ratio'],  mode='lines', name='Ratio of Token Edit Distance', line=dict(color="#e02430")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["idx"], y=df['ast_diff'],          mode='lines', name='AST Edit Distance', line=dict(color="#000000")), secondary_y=False)

    fig.update_xaxes(title_text="Index")
    fig.update_yaxes(range=[0, 30000], secondary_y=False)
    fig.update_yaxes(range=[0, 1], secondary_y=True)
    fig.update_layout(width=1200, height=800, title_text=f"Distances and Similarities on {name} Clones")

    fig.write_html(f"{basename}_{name}_line-plot.html")
    fig.write_image(f"{basename}_{name}_line-plot.png")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--test_predict_file', type=str, required=True)

    args = parser.parse_args()

    ## Validation each file exists
    args.model_name_or_path = os.path.dirname(args.test_predict_file)
    assert os.path.exists(args.model_name_or_path)
    assert os.path.exists(args.test_predict_file)


    ## Create logger
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO) 
    logger = logging.getLogger(__name__)

    ## Create test prediction dataframe
    test_res_df = pd.read_csv(args.test_predict_file, sep='\t')

    ## File path and name
    match_res = re.findall(r'(\.\.[a-zA-Z0-9\/\-\_]*)', args.test_predict_file)
    basename = match_res[0]
    
    similarity_df = create_similarity_table(df=test_res_df)
    similarity_df = similarity_df.sort_values(by=["char_diff"], ascending=False, ignore_index=True)
    
    ## Basic Statistics
    stats_df = basic_stats(df=similarity_df)
    stats_df.to_csv(f"{basename}_stats.tsv", index=False, sep='\t')
    
    tp_df = similarity_df[similarity_df["true_positive"]==1]
    fp_df = similarity_df[similarity_df["false_positive"]==1]
    fn_df = similarity_df[similarity_df["false_negative"]==1]
    tn_df = similarity_df[similarity_df["true_negative"]==1]

    ## Plot line diagram
    for df, name in zip([tp_df, fp_df, fn_df, tn_df], ["TP", "FP", "FN", "TN"]):
        plot_all_line_diagram(df=df, basename=basename, name=name)
    
    return None    

if __name__ == '__main__':
    main()
