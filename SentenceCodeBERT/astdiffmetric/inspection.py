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

def getPaths(pathName:str) -> list:
    condition = f'{pathName}/backSeqDel/*.jsonl'
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
            
        sourceSizeChar = len(line["source"])
        if sourceSizeChar <= upperSize:
            tokenLen = len(tokenizer.tokenize(line["source"]))
            if tokenLen <= 510:
                ranking[path] = tokenLen
    
    return max(ranking, key=ranking.get)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--upper_char_size', type=int)

    args = parser.parse_args()

    ## Load fine-tuned model
    model_tuned = SentenceTransformer(args.model_path)

    allJsonl = getPaths(args.test_data)

    longestCharPaths = logestCharCond(jsonlPaths=allJsonl, upperSize=args.upper_char_size)
    logging.info(longestCharPaths)
    
    realName = longestCharPaths.split("/")[-1].replace("back_del_", "")

    condition = f'{args.test_data}/*/*{realName}'
    fourTypeDataPath = glob.glob(condition, recursive=True)

    for path in fourTypeDataPath:
        with open(path) as f:
            jsonline = [json.loads(l) for l in f.readlines()]

        ## Embedding & calculate cosine simillarity
        for line in jsonline:
            source_embedding = model_tuned.encode(line["source"], convert_to_tensor=True)
            target_embedding = model_tuned.encode(line["target"], convert_to_tensor=True)

            cosine_score = util.cos_sim(source_embedding, target_embedding)
            line["cosSimInspect"] = cosine_score[0][0].item()
            
        df = pd.DataFrame(jsonline)

        ## Save data to jsonl & csv file
        basename = path.split("/")[-1].split(".")[0]
        storeCsvPath = f'{args.output_dir}/{basename}.csv'
        df.to_csv(storeCsvPath, index=False)

        logging.info(f"Saved! -> {storeCsvPath}.csv")


if __name__ == '__main__':
    main()
