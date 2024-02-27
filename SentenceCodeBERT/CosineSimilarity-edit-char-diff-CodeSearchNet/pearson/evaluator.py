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
from datetime import datetime
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers import models, losses, evaluation

import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


def correlation_coefficient(x: pd.DataFrame, y: pd.DataFrame) -> float:
    assert len(x) == len(y)
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = math.sqrt(((x - x_mean)**2).sum() * ((y - y_mean)**2).sum())
    return numerator / denominator

def draw_scatter_plot(df: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame, partition: str, basename: str) -> None:
    assert len(x) == len(y)
    fig = px.scatter(df, x="edit_distance", y="cosine_score", trendline="ols", color="pruning_type")
    fig.write_html(basename + "_eval-pearson-plot_{}.html".format(partition))
    fig.write_image(basename + "_eval-pearson-plot_{}.png".format(partition))
    return None

def main() -> None:
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--test_result_file', type=str, required=True)

    args = parser.parse_args()

    ## Validation each file exists
    assert os.path.exists(args.test_result_file)
    args.model_name_or_path = os.path.dirname(args.test_result_file)
    assert os.path.exists(args.model_name_or_path)


    ## Create logger
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO) 
    logger = logging.getLogger(__name__)

    ## Create test result dataframe
    test_res_df = pd.read_csv(args.test_result_file, sep='\t')
    sequence_df = test_res_df[test_res_df['pruning_type'] == "sequence"]
    subtree_df = test_res_df[test_res_df['pruning_type'] == "subtree"]

    ## File path and name
    match_res = re.findall(r'(\.\.[a-zA-Z0-9\/\-\_]*)', args.test_result_file)
    basename = match_res[0]
    
    ## Create plot figure
    ## All data
    draw_scatter_plot(test_res_df, test_res_df['edit_distance'], test_res_df['cosine_score'], "all", basename)
    ## sequence data
    draw_scatter_plot(sequence_df, sequence_df['edit_distance'], sequence_df['cosine_score'], "sequence", basename)
    ## subtree data
    draw_scatter_plot(subtree_df, subtree_df['edit_distance'], subtree_df['cosine_score'], "subtree", basename)

    eval_res = []
    for df, partition in zip((test_res_df, sequence_df, subtree_df), ("all", "sequence", "subtree")):
        data_size = len(df)
        pearson_score = correlation_coefficient(df['edit_distance'], df['cosine_score'])
        eval_res.append([partition, data_size, pearson_score])
    res_df = pd.DataFrame(eval_res, columns=["partition", "data_size", "pearson_score"])
    res_df.to_csv("{}.tsv".format(basename + "_eval-pearson_score"), sep='\t')

    return None    

if __name__ == '__main__':
    main()