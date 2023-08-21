import os
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

from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


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
        logging.info(f"<<< Total learning time : {time_diff} >>>")
        return res
    return _wrapper

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True)

def min_data_len(posi, nega):
    posi_size = len(posi)
    nega_size = len(nega)

    if posi_size < nega_size:
        return posi_size
    else:
        return nega_size

def create_inputs(args:argparse, jsonl_dir:str, partition:str, node_type:str) -> Dataset:
    logging.info(f"-- Create Training Dataset for {node_type} --")

    each_data_path = os.path.join(jsonl_dir, f"{partition}.jsonl")

    with open(each_data_path) as f:
        jsonl_data = [json.loads(l) for l in f.readlines()]

    ## Assort POSITIVE & NEGATIVE label data, and analysis statistics
    positive_jsonl = []
    negative_jsonl = []
    for line in jsonl_data:
        if line[node_type] == 1:
            positive_jsonl.append(line)
        elif line[node_type] == 0:
            negative_jsonl.append(line)
    
    logging.info(f"POSITIVE label dataset : {len(positive_jsonl)}")
    logging.info(f"NEGATIVE label dataset : {len(negative_jsonl)}")

    min_dataset_size = min_data_len(positive_jsonl, negative_jsonl)
        
    if min_dataset_size < int(args.use_dataset_size / 2):
        HALF_USE_SIZE = min_dataset_size
    else:
        HALF_USE_SIZE = int(args.use_dataset_size / 2)
        
        
    partition_dataset = []
    for posi_nega_jsonl in [positive_jsonl, negative_jsonl]:
        for num, _ in enumerate(tqdm(range(HALF_USE_SIZE))):
            line = {}
            random_line = random.choice(posi_nega_jsonl)
            line['idx'] = num
            line['sentence'] = random_line['text']
            line['label'] = random_line[node_type]
            partition_dataset.append(line)

    df = pd.DataFrame(partition_dataset)
    train_dataset = Dataset.from_pandas(df)
    encoded_dataset = train_dataset.map(preprocess_function, batched=True)
    logging.info(f'\n[The {partition} Dataset]\n{encoded_dataset}')

    return encoded_dataset


@funcspeed
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')

    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--do_pred', action="store_true")

    parser.add_argument('--node_types', nargs="*")
    parser.add_argument('--language', type=str)
    parser.add_argument('--use_dataset_size', type=int)

    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--evaluate_step', type=int, default=100)

    parser.add_argument('--save_model_base_dir', type=str)
    parser.add_argument('--input_base_dir', type=str)

    args = parser.parse_args()

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    if args.do_train:
        partition = "train"

        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_save_base_dir = os.path.join(args.save_model_base_dir, args.language, date_time)
        os.makedirs(model_save_base_dir, exist_ok=True)

        for node_type in args.node_types:
            logging.info(f"=== Fine-tuning for {node_type} ===")


            ## Dataset settings
            logging.info("Load Training Dataset from Json Line...")

            id2label = {0: "NEGATIVE", 1: "POSITIVE"}
            label2id = {"NEGATIVE": 0, "POSITIVE": 1}

            train_data = create_inputs(args, 
                                        os.path.join(args.input_base_dir, args.language), 
                                        partition,
                                        node_type)

            eval_data = create_inputs(args, 
                                        os.path.join(args.input_base_dir, args.language), 
                                        "valid",
                                        node_type)

            ## model settings
            logging.info("Load Pre-trained Model...")

            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, 
                                                                        num_labels=len(label2id),
                                                                        id2label=id2label, 
                                                                        label2id=label2id)

            # Train the model
            model_save_dir = os.path.join(model_save_base_dir, 
                                        args.model_name_or_path.replace("/", "-")+'_' + \
                                        node_type + '_' + \
                                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )

            logging_dir = os.path.join(model_save_dir, "log_tb")

            os.makedirs(model_save_dir)
            os.makedirs(logging_dir)

            train_args = TrainingArguments(
                    output_dir=model_save_dir, 
                    logging_dir=logging_dir,
                    evaluation_strategy = "epoch",
                    save_strategy = "epoch",
                    learning_rate=args.learning_rate,
                    per_device_train_batch_size=args.train_batch_size,
                    per_device_eval_batch_size=args.train_batch_size,
                    num_train_epochs=args.epoch_num,
                    weight_decay=args.weight_decay,
                    load_best_model_at_end=True,
                    )

            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=train_data,
                eval_dataset=eval_data,
                tokenizer=tokenizer,
            )

            logging.info("Start training...")
            trainer.train()
            logging.info("Finished to training model!")

    if args.do_eval:
        partition = "valid"

        for node_type in args.node_types:
            logging.info(f"=== Fine-tuning for {node_type} ===")


            ## Dataset settings
            logging.info("Load Training Dataset from Json Line...")

            id2label = {0: "NEGATIVE", 1: "POSITIVE"}
            label2id = {"NEGATIVE": 0, "POSITIVE": 1}

            valid_data = create_inputs(tokenizer, 
                                        os.path.join(args.input_base_dir, args.language), 
                                        partition,
                                        node_type)

            trainer.evaluate()
    
    if args.do_pred:
        partition = "test"
        pass

if __name__ == '__main__':
    main()
