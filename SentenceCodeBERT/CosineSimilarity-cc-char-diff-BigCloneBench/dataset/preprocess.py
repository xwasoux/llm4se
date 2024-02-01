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

def main() -> None:
    index_data = pd.read_json(os.path.join(".", "data.jsonl"), lines=True, orient='records', encoding='utf-8').set_index('idx')

    print("*** Train ***")
    train_df = pd.read_csv("train.txt", sep="\t", header=None, names=['idx1', 'idx2', 'label'])
    train_df = convert_edit_distance_parallel(index_data=index_data, pair_data=train_df)
    train_df.to_csv("train.txt", sep="\t")

    print("*** Valid ***")
    valid_df = pd.read_csv("valid.txt", sep="\t", header=None, names=['idx1', 'idx2', 'label'])
    valid_df = convert_edit_distance_parallel(index_data=index_data, pair_data=valid_df)
    valid_df.to_csv("valid.txt", sep="\t")

    print("*** Test ***")
    test_df = pd.read_csv("test.txt", sep="\t", header=None, names=['idx1', 'idx2', 'label'])
    test_df = convert_edit_distance_parallel(index_data=index_data, pair_data=test_df)
    test_df.to_csv("test.txt", sep="\t")

    return None

if __name__ == "__main__":
    main()
