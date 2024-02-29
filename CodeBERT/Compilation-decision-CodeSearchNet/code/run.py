import os
import sys
import csv
import glob
import gzip
import math
import pytz
import json
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

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import datasets
from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import evaluate

import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

def preprocess_function(examples, tokenizer: AutoTokenizer) -> dict:
    return tokenizer(examples["sentence"], truncation=True, padding=True)

def convert_idx_to_input_example(args: argparse.Namespace, tokenizer: AutoTokenizer, index_data: pd.DataFrame, context_data: pd.DataFrame, max_examples_size: int = None) -> datasets.Dataset:
    input_examples = []
    if max_examples_size is not None:
        positive_df = context_data[context_data["compilable"] == 1]
        negative_df = context_data[context_data["compilable"] == 0]
        positive_df = positive_df.sample(n=max_examples_size//2, random_state=42)
        negative_df = negative_df.sample(n=max_examples_size//2, random_state=42)
        context_data = pd.concat([positive_df, negative_df], ignore_index=True)

    id = 0
    for _, row in tqdm(context_data.iterrows(), total=len(context_data)):
        idx = row["idx"]
        text = index_data.loc[idx]["func"]
        compilable_label = row["compilable"]
        input_examples.append([id, text, compilable_label])
        id += 1
    input_examples = pd.DataFrame(input_examples, columns=["id", "sentence", "label"])
    input_examples = Dataset.from_pandas(input_examples)
    input_examples = input_examples.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    return input_examples

def load_and_cache_examples(args: argparse.Namespace, tokenizer: AutoTokenizer, eval: bool = False, test: bool = False, max_examples_size: int = None) -> datasets.Dataset:
    file_path = args.test_data_file if test else (args.valid_data_file if eval else args.train_data_file)
    context_data = pd.read_csv(file_path, sep="\t")
    index_data = pd.read_json(args.index_data_file, lines=True, orient="records", encoding="utf-8").set_index("idx")
    return convert_idx_to_input_example(args, tokenizer, index_data, context_data, max_examples_size)

def plot_learning_curve(model_save_path: str, partition: str, datetime_now: str) -> None:
    checkpoint_dirs = glob.glob(os.path.join(model_save_path, "checkpoint-*"))
    max_checkpoint_num = max([int(checkpoint_dir.split("-")[-1]) for checkpoint_dir in checkpoint_dirs])
    max_checkpoint_dir = os.path.join(model_save_path, "checkpoint-{}".format(max_checkpoint_num))
    
    with open(os.path.join(max_checkpoint_dir, "trainer_state.json"), "r") as f:
        train_state = json.load(f)
    
    loss_state = train_state["log_history"]
    loss_state_df = pd.DataFrame(loss_state).dropna(subset=["eval_loss"])
    fig = px.line(loss_state_df, x="epoch", y="eval_loss")

    fig.update_layout(
        title="Learning curve",
        xaxis_title="epoch",
        yaxis_title="eval_loss",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"))
    
    filename = os.path.join(max_checkpoint_dir, "{}_learning-curve_ckp-{}_{}.html".format(partition, max_checkpoint_num, datetime_now))
    fig.write_html(filename)
    fig.write_image(filename.replace(".html", ".png"))
    return None

def main() -> None:
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/codebert-base", required=True)
    parser.add_argument("--index_data_file", type=str, default=None, required=True)

    ## Other parameters
    parser.add_argument("--output_dir", type=str, default="./saved_models")
    parser.add_argument("--train_data_file", type=str, default=None)
    parser.add_argument("--valid_data_file", type=str, default=None)
    parser.add_argument("--test_data_file", type=str, default=None)
    parser.add_argument("--language", type=str, default=None)

    parser.add_argument("--train_max_examples_size", type=int)
    parser.add_argument("--valid_max_examples_size", type=int)
    parser.add_argument("--test_max_examples_size", type=int)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--evaluate_during_training", action="store_true")

    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--evaluate_step', type=int, default=100)

    args = parser.parse_args()

    torch.cuda.empty_cache()

    ## Create model and output directory
    if args.model_name_or_path == "microsoft/codebert-base":
        model_save_path = os.path.join(args.output_dir, 
                                        args.model_name_or_path.replace("/", "-") + "_" + \
                                        args.language + "_" + \
                                        datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        ## Load pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=len(label2id), 
                                                                    id2label=id2label, label2id=label2id)
    else:
        model_save_path = args.model_name_or_path
        if not os.path.exists(model_save_path):
            assert f"Model path not found: {model_save_path}. Please check the model path."
        ## Load fine-tuned model
        classifier = pipeline(task="sentiment-analysis", model=model_save_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    ## Create logger
    partition = "test" if args.do_test else ("valid" if args.do_eval else "train")
    datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = partition + "_logger_" + datetime_now
    logging.basicConfig(format="%(asctime)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO, 
                        filename=os.path.join(model_save_path, f"{log_filename}.txt"))
    logger = logging.getLogger(__name__)


    if args.do_train:
        ## Create data loader and loss function for ContrastiveLoss
        train_dataset = load_and_cache_examples(args, tokenizer, eval=False, test=False, max_examples_size=args.train_max_examples_size)
        from pprint import pprint 
        pprint(train_dataset, sort_dicts=False)
        print(type(train_dataset))
        ## Create data loader and evaluator for EmbeddingSimilarityEvaluator
        if args.evaluate_during_training:
            valid_dataset = load_and_cache_examples(args, tokenizer, eval=True, test=False, max_examples_size=args.valid_max_examples_size)

        ## Train the model
        logger.info("********** Running training **********")
        logger.info("   Num examples = {}".format(len(train_dataset)))
        logger.info("   Num labels = {}".format(len(label2id)))
        logger.info("       Positive examples = {}".format(len([label for label in train_dataset["label"] if label == 1])))
        logger.info("       Negative examples = {}".format(len([label for label in train_dataset["label"] if label == 0]))) 
        logger.info("   Num epoch = {}".format(args.epoch_num))
        logger.info("   Batch size = {}".format(args.train_batch_size))
        logger.info("   Evaluate step = {}".format(args.evaluate_step))
        logger.info("   Model save path: {}".format(model_save_path))

        train_args = TrainingArguments(
                output_dir=model_save_path, 
                evaluation_strategy = "epoch",
                save_strategy = "epoch",
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.train_batch_size,
                per_device_eval_batch_size=args.train_batch_size,
                num_train_epochs=args.epoch_num,
                weight_decay=args.weight_decay,
                load_best_model_at_end=True)

        if args.evaluate_during_training:
            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                tokenizer=tokenizer)

        else:
            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer)

        trainer.train()
        plot_learning_curve(model_save_path, partition, datetime_now)

    if args.do_eval:
        ## Create data loader and evaluator for EmbeddingSimilarityEvaluator
        valid_dataset = load_and_cache_examples(args, eval=True, test=False, max_examples_size=args.valid_max_examples_size)

        ## Evaluate the model
        logger.info("********** Running evaluation **********")
        logger.info("   Num examples = {}".format(len(valid_dataset)))
        logger.info("       Positive examples = {}".format(len([label for label in valid_dataset["label"] if label == 1])))
        logger.info("       Negative examples = {}".format(len([label for label in valid_dataset["label"] if label == 0]))) 
        logger.info("   Batch size = {}".format(args.train_batch_size))
        logger.info("   Model save path: {}".format(model_save_path))

        references = valid_dataset["label"]
        predictions = trainer.predict(valid_dataset)

        ## Evaluate the model
        accuracy_metric = evaluate.load("accuracy")
        presition_metric = evaluate.load("presition")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")
        
        accuracy = accuracy_metric.compute(predictions=predictions, references=references)
        presition = presition_metric.compute(predictions=predictions, references=references)
        recall = recall_metric.compute(predictions=predictions, references=references)
        f1 = f1_metric.compute(predictions=predictions, references=references)

        evaluate_score_tb = [accuracy, presition, recall, f1]
        evaluate_score_df = pd.DataFrame(evaluate_score_tb, index=["accuracy", "precision", "recall", "f1"])
        evaluate_score_df.to_csv(os.path.join(model_save_path, f"{partition}_eval-scores_{datetime_now}.tsv"), sep="\t")
        
    if args.do_test:
        ## Load test data and index data
        context_data = pd.read_csv(args.test_data_file, sep="\t")
        index_data = pd.read_json(args.index_data_file, lines=True, orient="records", encoding="utf-8").set_index("idx")

        ## Extract max_examples_size examples
        max_examples_size = args.test_max_examples_size
        if max_examples_size is not None:
            positive_df = context_data[context_data["compilable"] == 1]
            negative_df = context_data[context_data["compilable"] == 0]
            positive_df = positive_df.sample(n=max_examples_size//2, random_state=42)
            negative_df = negative_df.sample(n=max_examples_size//2, random_state=42)
            context_data = pd.concat([positive_df, negative_df], ignore_index=True)

        ## Evaluate the model
        logger.info("********** Running test **********")
        logger.info("   Num examples = {}".format(len(context_data)))
        logger.info("       Positive examples = {}".format(len(context_data[context_data["compilable"] == 1])))
        logger.info("       Negative examples = {}".format(len(context_data[context_data["compilable"] == 0])))
        logger.info("   Model save path: {}".format(model_save_path))

        ## Inferencing
        prediction_table = []
        for _, row in tqdm(context_data.iterrows(), total=len(context_data)):
            idx = row["idx"]
            text = index_data.loc[idx]["func"]

            pruning_type = row["pruning_type"]
            compilable_label = row["compilable"]

            ## Classifier
            outputs = classifier(text, truncation=True, padding=True)
            prediction = label2id[outputs[0]["label"]]

            prediction_table.append([idx, text, pruning_type, compilable_label, prediction])
        prediction_df = pd.DataFrame(prediction_table, columns=["idx", "text", "pruning_type", "compilable_label", "prediction"])
        filename = os.path.join(model_save_path, "{}.txt".format(partition + "-predict_" + datetime_now))
        prediction_df.to_csv(filename, sep="\t", index=False)
        

if __name__ == "__main__":
    main()
