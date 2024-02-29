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
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers import models, losses, evaluation

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

def extract_base_name(file_path: str) -> str:
    match_res = re.findall(r'(\.\.[a-zA-Z0-9\/\-\_]*)', file_path)
    return match_res[0]

def compute_confusion_matrix(df: pd.DataFrame, actual: str, predict: str) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true=df[actual], y_pred=df[predict]).ravel()
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

def compute_metrics(df: pd.DataFrame, actual: str, predict: str) -> Dict[str, float]:
    accuracy = accuracy_score(y_true=df[actual], y_pred=df[predict])
    precision = precision_score(y_true=df[actual], y_pred=df[predict])
    recall = recall_score(y_true=df[actual], y_pred=df[predict])
    f1 = f1_score(y_true=df[actual], y_pred=df[predict])
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def evaluate(df: pd.DataFrame, actual: str, predict: str) -> tuple:
    confusion_matrix = compute_confusion_matrix(df, actual, predict)
    metrics = compute_metrics(df, actual, predict)
    return (confusion_matrix, metrics)
    
def main() -> None:
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--test_predict_file', type=str, required=True)

    args = parser.parse_args()

    ## Validation each file exists
    assert os.path.exists(args.test_predict_file)

    ## Create logger
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO) 
    logger = logging.getLogger(__name__)

    ## Create test predict dataframe
    test_pred_df = pd.read_csv(args.test_predict_file, sep='\t')

    ## File path and name
    basename = extract_base_name(args.test_predict_file)

    thresholds_list = list(np.round(np.arange(0.1, 1.0, 0.05), 2))

    confusion_matrix_table = {}
    metrics_table = {}
    for threshold in thresholds_list:
        ## Compute confusion matrix and metrics
        test_pred_df[f"predict_{threshold}"] = test_pred_df['cosine_score'] > threshold
        confusion_matrix, metrics = evaluate(test_pred_df, "clone", f"predict_{threshold}")

        ## Save confusion matrix and metrics
        confusion_matrix_table[f"threshold-{threshold}"] = confusion_matrix
        metrics_table[f"threshold-{threshold}"] = metrics

    ## Save confusion matrix and metrics
    confusion_matrix_df = pd.DataFrame(confusion_matrix_table).T
    confusion_matrix_df.to_csv(f"{basename}_confusion_matrix.tsv", sep='\t')
    
    metrics_df = pd.DataFrame(metrics_table).T
    metrics_df.to_csv(f"{basename}_metrics.tsv", sep='\t')

    ## Save all prediction
    test_pred_df.to_csv(f"{basename}_predicts-table.tsv", sep='\t')
    
    ## Draw ROC curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(test_pred_df['clone'], test_pred_df[f"cosine_score"])
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={roc_auc:.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=800, height=800
    )
    fig.write_html(f"{basename}_roc-curve.html")
    fig.write_image(f"{basename}_roc-curve.png")
    
    return None    

if __name__ == '__main__':
    main()
