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
    fig = px.scatter(df, x="char_diff", y="cosine_score", trendline="ols", color="label")
    fig.write_html(basename + "_eval-pearson-plot_{}.html".format(partition))
    fig.write_image(basename + "_eval-pearson-plot_{}.png".format(partition))
    return None

def main() -> None:
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--test_result_file', type=str, required=True)

    args = parser.parse_args()

    ## Validation each file exists
    assert os.path.exists(args.model_name_or_path)
    assert os.path.exists(args.test_result_file)


    ## Create logger
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO) 
    logger = logging.getLogger(__name__)

    ## Create test result dataframe
    test_res_df = pd.read_csv(args.test_result_file, sep='\t')
    positive_df = test_res_df[test_res_df['label'] == 1]
    negative_df = test_res_df[test_res_df['label'] == 0]

    ## File path and name
    match_res = re.findall(r'(\.\.[a-zA-Z0-9\/\-\_]*)', args.test_result_file)
    basename = match_res[0]
    
    ## Create plot figure
    ## All data
    draw_scatter_plot(test_res_df, test_res_df['char_diff'], test_res_df['cosine_score'], "all", basename)
    ## Positive data
    draw_scatter_plot(positive_df, positive_df['char_diff'], positive_df['cosine_score'], "positive", basename)
    ## Negative data
    draw_scatter_plot(negative_df, negative_df['char_diff'], negative_df['cosine_score'], "negative", basename)

    eval_res = []
    for df, partition in zip((test_res_df, positive_df, negative_df), ("all", "positive", "negative")):
        data_size = len(df)
        pearson_score = correlation_coefficient(df['char_diff'], df['cosine_score'])
        eval_res.append([partition, data_size, pearson_score])
    res_df = pd.DataFrame(eval_res, columns=["partition", "data_size", "pearson_score"])
    res_df.to_csv("{}.tsv".format(basename + "_eval-pearson_score"), sep='\t')

    return None    

if __name__ == '__main__':
    main()