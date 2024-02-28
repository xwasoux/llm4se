import os
import sys
import csv
import ast
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
from typing import List, Dict, Any, Tuple
from astars import AParser, APruner

def judge_compilable_code(args: argparse.Namespace, code: str) -> int:
    try:
        if args.language == "python":
            ast.parse(code)
        return 1
    except:
        return 0

def create_edit_code_judge(args_func: Tuple[argparse.Namespace, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    args, func_list = args_func
    candidate_functions = [(row["repo"], row["path"], row["func_name"], row["code"]) for idx, row in func_list.iterrows()]
    parser = AParser(lang=args.language)

    edit_judge = []
    edit_func = []
    for repo, path, func_name, original_code in tqdm(candidate_functions):
        if original_code is None or len(original_code) == 0:
            continue
        try:
            original_code = parser.preprocess(original_code)
            parse_tree = parser.parse(original_code)
        except:
            continue

        repo = repo.replace("/", "-")
        path = path.replace("/", "-")
        original_code_idx = "{}={}={}".format(repo, path, func_name)

        original_code_tree = str(parse_tree)
        edit_func.append([original_code_idx, original_code, original_code_tree])
        edit_judge.append([original_code_idx, "original", judge_compilable_code(args=args, code=original_code)])

        sequence_res = APruner.sequencialBackwardPrune(tree=parse_tree)
        subtree_res = APruner.sequencialSubtreePrune(tree=parse_tree)

        idx = 1
        for east in sequence_res:
            east_code = east[0].recover()
            if east_code == "":
                continue
            edit_code_idx = f"{repo}={path}={func_name}={idx:04d}"
            edit_judge.append([edit_code_idx, "sequence", judge_compilable_code(args=args, code=east_code)])
            east_code_tree = str(east[0])
            edit_func.append([edit_code_idx, east_code, east_code_tree])
            idx += 1

        for east in subtree_res:
            east_code = east[0].recover()
            if east_code == "":
                continue
            edit_code_idx = f"{repo}={path}={func_name}={idx:04d}"
            edit_judge.append([edit_code_idx, "subtree", judge_compilable_code(args=args, code=east_code)])
            east_code_tree = str(east[0])
            edit_func.append([edit_code_idx, east_code, east_code_tree])
            idx += 1

    edit_judge_df = pd.DataFrame(edit_judge, columns=["idx", "pruning_type", "compilable"])
    edit_func_df = pd.DataFrame(edit_func, columns=["idx", "func", "tree"])
    return edit_judge_df, edit_func_df

def filtered_func_list(func_list: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    func_list = func_list[func_list["code"].apply(lambda x: len(x) > 0)]
    func_list = func_list[func_list["code"].apply(lambda x: len(x) <= args.max_string_length)]
    return func_list

def do_preprocess(args: argparse.Namespace, eval: bool = False, test: bool = False) -> pd.DataFrame:
    input_file = args.test_data_file if test else (args.valid_data_file if eval else args.train_data_file)
    output_file = args.output_test_data_file if test else (args.output_valid_data_file if eval else args.output_train_data_file)

    data_size = args.test_size if test else (args.valid_size if eval else args.train_size)
    func_list = pd.read_json(input_file, lines=True)
    func_list = filtered_func_list(func_list, args)
    if args.size_all is False:
        func_list = func_list.sample(n=data_size, random_state=args.seed)

    # Split the data for parallel processing
    num_processes = os.cpu_count()  # Use the number of available CPU cores
    func_list_split = np.array_split(func_list, num_processes)

    # Use Pool for parallel processing
    with Pool(num_processes) as pool:
        args_func_list = [(args, split) for split in func_list_split]
        results = pool.map(create_edit_code_judge, args_func_list)

    # Combine results from parallel processes
    edit_judge_dfs, edit_func_dfs = zip(*results)
    edit_judge_df = pd.concat(edit_judge_dfs, ignore_index=True)
    edit_func_df = pd.concat(edit_func_dfs, ignore_index=True)

    edit_judge_df.to_csv(output_file, sep="\t", index=False)
    return edit_func_df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", help="Language of the dataset")
    parser.add_argument("--train_data_file", type=str, default=None, help="Path to the training data file")
    parser.add_argument("--valid_data_file", type=str, default=None, help="Path to the validation data file")
    parser.add_argument("--test_data_file", type=str, default=None, help="Path to the test data file")

    parser.add_argument("--output_train_data_file", type=str, default=None, help="Path to the output training data file")
    parser.add_argument("--output_valid_data_file", type=str, default=None, help="Path to the output validation data file")
    parser.add_argument("--output_test_data_file", type=str, default=None, help="Path to the output test data file")

    parser.add_argument("--size_all", action="store_true", help="Max size of the dataset")
    parser.add_argument("--train_size", type=int, default=10000, help="Size of the training dataset")
    parser.add_argument("--valid_size", type=int, default=8000, help="Size of the validation dataset")
    parser.add_argument("--test_size", type=int, default=8000, help="Size of the test dataset")

    parser.add_argument("--max_string_length", type=int, default=1000, help="Limit the maximum length of the string to embedding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    if args.train_data_file is not None or args.valid_data_file is not None or args.test_data_file is not None:
        assert "Please do not provide the path to the dataset files"

    if os.path.exists(os.path.join(".", "CodeSearchNet", args.language)):
        args.train_data_file = os.path.join(".", "CodeSearchNet", args.language, "train.jsonl")
        args.valid_data_file = os.path.join(".", "CodeSearchNet", args.language, "valid.jsonl")
        args.test_data_file = os.path.join(".", "CodeSearchNet", args.language, "test.jsonl")
        args.output_train_data_file = os.path.join(".", f"{args.language}-ed-train.txt")
        args.output_valid_data_file = os.path.join(".", f"{args.language}-ed-valid.txt")
        args.output_test_data_file = os.path.join(".", f"{args.language}-ed-test.txt")
    else:
        assert "Please provide the path to the dataset files"

    print("*** Train ***")
    train_func_data = do_preprocess(args, eval=False, test=False)

    print("*** Valid ***")
    valid_func_data = do_preprocess(args, eval=True, test=False)

    print("*** Test ***")
    test_func_data = do_preprocess(args, eval=False, test=True)

    # Concat all the data and save
    all_func_data = pd.concat([train_func_data, valid_func_data, test_func_data])
    all_func_data.to_json(os.path.join(".", f"{args.language}-ed-data.jsonl"), orient="records", lines=True)
    return None

if __name__ == "__main__":
    main()
