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

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def evaluate(df: pd.DataFrame) -> tuple:
    tn, fp, fn, tp = confusion_matrix(y_true=df["clone_label"], y_pred=df["prediction"]).ravel()
    confusion_matrix_tb = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    accuracy = accuracy_score(y_true=df["clone_label"], y_pred=df["prediction"])
    precision = precision_score(y_true=df["clone_label"], y_pred=df["prediction"])
    recall = recall_score(y_true=df["clone_label"], y_pred=df["prediction"])
    f1 = f1_score(y_true=df["clone_label"], y_pred=df["prediction"])
    
    return (confusion_matrix_tb, [accuracy, precision, recall, f1])

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
    
    prediction = evaluate(test_res_df)
    confusion_matrix_tb = prediction[0]
    evaluate_scores = prediction[1]

    confusion_matrix_df = pd.DataFrame(confusion_matrix_tb, index=[0])
    confusion_matrix_df.to_csv("{}.tsv".format(basename + "_eval-confusion_matrix"), sep='\t')
    score_df = pd.DataFrame(evaluate_scores, index=["accuracy", "precision", "recall", "f1"])
    score_df.to_csv("{}.tsv".format(basename + "_eval-scores"), sep='\t')

    return None    

if __name__ == '__main__':
    main()
