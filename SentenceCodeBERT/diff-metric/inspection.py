import os
import re
import sys
import csv
import json
import glob
import gzip
import math
import random
import logging
import argparse
import pickle
import pandas as pd
from datetime import datetime
from tqdm import tqdm, trange

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler


from transformers import AutoTokenizer

from tensorboardX import SummaryWriter

from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def get_children_dir_paths(path_name:str) -> list:
    condition = f'{path_name}/*/'
    return glob.glob(condition, recursive=True)

def get_jonsl_paths(path_name:str) -> list:
    condition = f'{path_name}/*.jsonl'
    return glob.glob(condition, recursive=True)

def logest_char_cond(jsonl_paths:list, upper_size:int) -> str:
    ranking = {}
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    for path in jsonl_paths:
        with open(path) as f:
            try:
                line = json.loads(f.readline())
            except:
                continue
            
        original_code = line["cleaned_code"]
        source_size_char = len(original_code)
        if source_size_char <= upper_size:
            token_len = len(tokenizer.tokenize(original_code))
            if token_len <= 510:
                ranking[path] = token_len
    
    return max(ranking, key=ranking.get)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', nargs='*')

    parser.add_argument('--test_base_dir', type=str)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--test_data', nargs='*')
    
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()

    test_data_paths = []
    for pruning_type in args.test_data:
        test_jsonl_path = os.path.join(args.test_base_dir, pruning_type)
        each_test_data = get_jonsl_paths(test_jsonl_path)
        test_data_paths.extend(each_test_data)

    longest_char_paths = logest_char_cond(jsonl_paths=test_data_paths, upper_size=800)
    logging.info(longest_char_paths)

    with open(longest_char_paths) as f:
        jsonline = [json.loads(l) for l in f.readlines()]

    ## Embedding & calculate cosine simillarity
    pooler_types = []
    for model_name in args.model_path:
        ## Load fine-tuned model
        model_tuned = SentenceTransformer(model_name)
        
        res = re.search(r'(cls|max|mean)', model_name)
        pooler_name = res.group()
        pooler_types.append(pooler_name)
        for line in jsonline:
            source_embedding = model_tuned.encode(line["cleaned_code"], convert_to_tensor=True)
            target_embedding = model_tuned.encode(line["edited_code"], convert_to_tensor=True)

            cosine_score = util.cos_sim(source_embedding, target_embedding)
            line[f"inspect_{pooler_name}"] = cosine_score[0][0].item()
        
    df = pd.DataFrame(jsonline)

    ## Save data to jsonl & csv file
    basename = longest_char_paths.split("/")[-1].split(".")[0]
    store_csv_path = os.path.join(args.output_path, basename)
    os.makedirs(store_csv_path, exist_ok=True)
    
    time_label = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_label = "-".join(args.test_data)
    pooler_label = "-".join(pooler_types)
    store_csv = os.path.join(store_csv_path, f"{time_label}_{data_label}_{pooler_label}.csv")
    
    df.to_csv(store_csv, index=False)
    logging.info(f"Saved! -> {store_csv}")


if __name__ == '__main__':
    main()
