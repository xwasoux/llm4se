import os
import re
import sys
import csv
import glob
import gzip
import math
import json
import time
import random
import logging
import argparse
import pickle
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
from torch import nn
import tensorboard
from tensorboardX import SummaryWriter

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from datasets import Dataset, DatasetDict, load_metric
from transformers import pipeline


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
                    # handlers=[LoggingHandler()])

def funcspeed(func):
    def _wrapper(*args, **keywargs):
        start_time = datetime.today()
        res = func(*args, **keywargs)
        end_time = datetime.today()
        time_diff = end_time - start_time
        logging.info(f"<<< Total processing time : {time_diff} >>>")
        return res
    return _wrapper

def get_dir_paths(dir_path:str) -> list:
    condition = f'{dir_path}/*/'
    return glob.glob(condition, recursive=True)

def get_jsonl_paths(dir_path:str) -> list:
    condition = f'{dir_path}/*.jsonl'
    return glob.glob(condition, recursive=True)

def load_test_data(jsonl_dir:str, partition:str) -> list:
    logging.info("-- Loading Test Data --")

    each_data_path = os.path.join(jsonl_dir, f"{partition}.jsonl")

    with open(each_data_path) as f:
        jsonl_data = [json.loads(l) for l in f.readlines()]

    return jsonl_data

def get_model_dir(model_dirs:str, node_type:str) -> str:
    model_dirs = get_dir_paths(model_dirs)
    
    for path in model_dirs:
        match_res = re.search(r'[a-z]+(_statement|_clause)', str(path))
        if match_res != None:
            model_base_dir = path
            continue
        else:
            pass
    each_checkpoints = get_dir_paths(model_base_dir)
    checkpoint_num = {}
    for each_path in each_checkpoints:
        match_res = re.search(r'checkpoint-[0-9]+', each_path)
        if match_res != None:
            chpt_dir = match_res.group()
            chpt_num = chpt_dir.split("-")[-1]
            checkpoint_num[chpt_dir] = int(chpt_num)
    max_num_dir = max(checkpoint_num, key=checkpoint_num.get)
    return os.path.join(model_base_dir, max_num_dir)

def evaluate_model(dataframe:pd.DataFrame, node_types:list) -> list:
    return_jsonl = []
    for each_type in node_types:
        each_res = {}

        actual_list = dataframe[each_type].to_list()
        predict_list = dataframe[f"{each_type}-inference"].to_list()

        each_res["node_type"] = each_type

        tn, fp, fn, tp = confusion_matrix(actual_list, predict_list).ravel()
        each_res["true_negative"] = tn
        each_res["false_positive"] = fp
        each_res["false_negative"] = fn
        each_res["true_positive"] = tp

        each_res["accuracy"] = accuracy_score(actual_list, predict_list)
        each_res["precision"] = precision_score(actual_list, predict_list)
        each_res["recall"] = recall_score(actual_list, predict_list)
        each_res["f1_score"] = f1_score(actual_list, predict_list)

        return_jsonl.append(each_res)

    return return_jsonl



@funcspeed
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--node_types', nargs="*")
    parser.add_argument('--language', type=str)

    parser.add_argument('--input_base_dir', type=str)
    parser.add_argument('--pretrained_model_base_dir', type=str)
    parser.add_argument('--output_base_dir', type=str)

    args = parser.parse_args()

    ''' 
    [coding perspective]
    
    load test data from jsonl
    
    for each_statement in node_types:
        load fine-tuned model

        for line in jsonl:
            extract code
        
            inference 

            append inference label
            append inference score
        
    transform data from jsonl to dataframe
    save to csv
    
    '''

    
    partition = "test"
    test_jsonl_data = load_test_data(os.path.join(args.input_base_dir, args.language), partition)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    logging.info(f"=== Random Test Data Sampling ===")
    USE_SIZE = 2500
    inference_jsonl = random.sample(test_jsonl_data, USE_SIZE)
    logging.info(f"Test Data size: {len(inference_jsonl)}")

    for node_type in args.node_types:
        logging.info(f"=== Inference for {node_type} ===")

        tuned_model_dir = get_model_dir(model_dirs=os.path.join(args.pretrained_model_base_dir, args.language),
                                        node_type=node_type)

        for line in tqdm(inference_jsonl):
            text = line["text"]
            classifier = pipeline(task="sentiment-analysis", model=tuned_model_dir)
            classificate_res = classifier(text)

            inf_res = label2id[classificate_res[0]["label"]]
            line[f"{node_type}-inference"] = inf_res
            line[f"score{node_type}-score"] = classificate_res[0]["score"]
            line[f"{node_type}-judge"] = "CORRECT" if inf_res == line[node_type] else "INCORRECT"
    #     break
    # return
    
    df = pd.DataFrame(inference_jsonl)

    time_label = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_inf_table_dir = os.path.join(args.output_base_dir, args.language, time_label)
    os.makedirs(save_inf_table_dir, exist_ok=True)
    df.to_csv(os.path.join(save_inf_table_dir, 
                           f"inference_random_chice_{USE_SIZE}.csv"),index=False)

    ## Calculate Accuracy
    eval_jsonl = evaluate_model(dataframe=df, node_types=args.node_types)

    eval_df = pd.DataFrame(eval_jsonl)
    eval_df.to_csv(os.path.join(save_inf_table_dir, 
                                f"evaluate_random_chice_{USE_SIZE}.csv"),index=False)

               

if __name__ == '__main__':
    main()
