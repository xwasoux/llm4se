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
from tensorboardX import SummaryWriter

from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--datasets', type=str)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    ## Load fine-tuned model
    model_tuned = SentenceTransformer(args.model_path)

    with open(args.datasets) as f:
        dictl_data = [json.loads(l) for l in f.readlines()]
    
    ## Embedding & calculate cosine simillarity
    for line in dictl_data:
        source_embedding = model_tuned.encode(line["source"], convert_to_tensor=True)
        target_embedding = model_tuned.encode(line["target"], convert_to_tensor=True)

        cosine_score = util.cos_sim(source_embedding, target_embedding)
        line["cos_sim_inspect"] = cosine_score[0][0].item()
        
    df = pd.DataFrame(dictl_data)

    ## Save data to jsonl & csv file
    base_name = args.datasets.split("/")[-1].split(".")[0]
    df.to_json(f'{args.model_path}/{base_name}.jsonl', orient='records', force_ascii=False, lines=True)
    df.to_csv(f'{args.model_path}/{base_name}.csv', index=False)

    logging.info(f"Saved! -> {args.model_path}/{base_name}.jsonl")
    logging.info(f"Saved! -> {args.model_path}/{base_name}.csv")


if __name__ == '__main__':
    main()
