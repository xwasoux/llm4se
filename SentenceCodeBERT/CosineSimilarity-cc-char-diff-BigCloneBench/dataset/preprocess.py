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
from multiprocessing import Pool

def distance_to_cosine(distance: int) -> float:
    return 1 / (1 + distance)

def convert_edit_distance_helper(args):
    index_data, row = args
    idx1 = row['idx1']
    idx2 = row['idx2']
    label = row['label']
    text1 = index_data.loc[idx1]['func']
    text2 = index_data.loc[idx2]['func']

    edit_distance = Levenshtein.distance(text1, text2)
    cosine_label = distance_to_cosine(edit_distance) if label == 1 else -1

    return idx1, idx2, edit_distance, cosine_label

def convert_edit_distance_parallel(index_data: pd.DataFrame, pair_data: pd.DataFrame) -> pd.DataFrame:
    with Pool() as pool:
        args_list = [(index_data, row) for _, row in pair_data.iterrows()]
        results = list(tqdm(pool.imap(convert_edit_distance_helper, args_list), total=len(pair_data)))

    for result, (_, row) in zip(results, pair_data.iterrows()):
        idx1, idx2, edit_distance, cosine_label = result
        pair_data.loc[row.name, 'edit_distance'] = edit_distance
        pair_data.loc[row.name, 'cosine_label'] = cosine_label

    return pair_data

def do_preprocess(args: argparse.Namespace, index_data: pd.DataFrame, eval: bool = False, test: bool = False) -> None:
    input_file = args.test_data_file if test else (args.valid_data_file if eval else args.train_data_file)
    output_file = args.output_test_data_file if test else (args.output_valid_data_file if eval else args.output_train_data_file)
    
    data_size = args.test_size if test else (args.valid_size if eval else args.train_size)
    if args.size_unified:
        data_size = args.size_unified
    elif args.size_unspecified:
        data_size = None
    elif data_size:
        pass
    
    df = pd.read_csv(input_file, sep="\t", header=None, names=['idx1', 'idx2', 'label'])
    if data_size:
        positive_df = df[df['label'] == 1].sample(n=data_size // 2, random_state=args.seed)
        negative_df = df[df['label'] == 0].sample(n=data_size // 2, random_state=args.seed)
        df = pd.concat([positive_df, negative_df])

    df = convert_edit_distance_parallel(index_data=index_data, pair_data=df)
    df.to_csv(output_file, sep="\t")

    return None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--func_data", type=str, default="data.jsonl", help="Path to the data file")
    parser.add_argument("--train_data_file", type=str, default="train.txt", help="Path to the training data file")
    parser.add_argument("--valid_data_file", type=str, default="valid.txt", help="Path to the validation data file")
    parser.add_argument("--test_data_file", type=str, default="test.txt", help="Path to the test data file")
    
    parser.add_argument("--output_train_data_file", type=str, default="ed-train.txt", help="Path to the output training data file")
    parser.add_argument("--output_valid_data_file", type=str, default="ed-valid.txt", help="Path to the output validation data file")
    parser.add_argument("--output_test_data_file", type=str, default="ed-test.txt", help="Path to the output test data file")
    
    parser.add_argument("--size_unified", type=int, help="Unified size of the dataset")
    parser.add_argument("--size_unspecified", action="store_true", help="Unified size of the dataset")
    parser.add_argument("--train_size", type=int, default=80000, help="Size of the training dataset")
    parser.add_argument("--valid_size", type=int, default=10000, help="Size of the validation dataset")
    parser.add_argument("--test_size", type=int, default=10000, help="Size of the test dataset")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()
    
    index_data = pd.read_json(args.func_data, lines=True, orient='records', encoding='utf-8').set_index('idx')

    print("*** Train ***")
    do_preprocess(args, index_data, eval=False, test=False)

    print("*** Valid ***")
    do_preprocess(args, index_data, eval=True, test=False)

    print("*** Test ***")
    do_preprocess(args, index_data, eval=False, test=True)
    
    return None

if __name__ == "__main__":
    main()
