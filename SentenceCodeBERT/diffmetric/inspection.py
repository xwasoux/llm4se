import os
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

def getDirPaths(pathName:str) -> list:
    condition = f'{pathName}/*/'
    return glob.glob(condition, recursive=True)

def getJonslPaths(pathName:str) -> list:
    condition = f'{pathName}/*.jsonl'
    return glob.glob(condition, recursive=True)

def logestCharCond(jsonlPaths:list, upperSize:int) -> str:
    ranking = {}
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    for path in jsonlPaths:
        with open(path) as f:
            try:
                line = json.loads(f.readline())
            except:
                continue
            
        original_code = line["originalCode"]
        sourceSizeChar = len(original_code)
        if sourceSizeChar <= upperSize:
            tokenLen = len(tokenizer.tokenize(original_code))
            if tokenLen <= 510:
                ranking[path] = tokenLen
    
    return max(ranking, key=ranking.get)


def shortestCharCond(jsonlPaths:list) -> str:
    ranking = {}
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    for path in jsonlPaths:
        with open(path) as f:
            try:
                line = json.loads(f.readline())
            except:
                continue
            
        original_code = line["originalCode"]
        sourceSizeChar = len(original_code)
        if sourceSizeChar <= 300:
            tokenLen = len(tokenizer.tokenize(original_code))
            ranking[path] = tokenLen
    
    return min(ranking, key=ranking.get)

def cos_sim_inspect(path:str, model_tuned:SentenceTransformer) -> pd.DataFrame:
    with open(path) as f:
        jsonline = [json.loads(l) for l in f.readlines()]

    ## Embedding & calculate cosine simillarity
    for line in jsonline:
        source_embedding = model_tuned.encode(line["originalCode"], convert_to_tensor=True)
        target_embedding = model_tuned.encode(line["editedCode"], convert_to_tensor=True)

        cosine_score = util.cos_sim(source_embedding, target_embedding)
        line["cosSimInspect"] = cosine_score[0][0].item()
        
    return pd.DataFrame(jsonline)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str)

    parser.add_argument('--test_base_dir', type=str)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--test_data', nargs='*')
    
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    ## Load fine-tuned model
    model_tuned = SentenceTransformer(args.model_path)


    test_data_paths = []
    for delete_type in args.test_data:
        test_jsonl_path = os.path.join(args.test_base_dir, delete_type)
        each_test_data = getJonslPaths(test_jsonl_path)
        test_data_paths.extend(each_test_data)

    longestCharPath = logestCharCond(jsonlPaths=test_data_paths, upperSize=800)
    shortestCharPath = shortestCharCond(jsonlPaths=test_data_paths)
    logging.info(f"Longest Char Code: {longestCharPath}")
    logging.info(f"Shortest Char Code: {shortestCharPath}")
    
    for jsonlpath in [longestCharPath, shortestCharPath]:
        df = cos_sim_inspect(path=jsonlpath, model_tuned=model_tuned)

        ## Save data to jsonl & csv file
        basename = jsonlpath.split("/")[-1].split(".")[0]
        storeCsvPath = os.path.join(args.model_path, args.output_dir)
        os.makedirs(storeCsvPath, exist_ok=True)
        storeCsv = os.path.join(args.model_path, args.output_dir, f"{basename}.csv")
        df.to_csv(storeCsv, index=False)
        logging.info(f"Saved! -> {storeCsv}.csv")


if __name__ == '__main__':
    main()
