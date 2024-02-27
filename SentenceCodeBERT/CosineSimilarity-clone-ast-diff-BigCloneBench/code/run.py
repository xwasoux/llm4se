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

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers import models, losses, evaluation

def distance_to_cosine(distance: int) -> float:
    return 1/(1+distance)

def convert_idx_to_input_example(args: argparse, index_data: pd.DataFrame, pair_data: pd.DataFrame, max_examples_size: int = None) -> list:
    input_examples = []
    if max_examples_size is not None:
        positive_pair = pair_data[pair_data["label"] == 1]
        negative_pair = pair_data[pair_data["label"] == 0]
        positive_pair = positive_pair.sample(n=max_examples_size//2, random_state=42)
        negative_pair = negative_pair.sample(n=max_examples_size//2, random_state=42)
        pair_data = pd.concat([positive_pair, negative_pair], ignore_index=True)

    for _, row in tqdm(pair_data.iterrows(), total=len(pair_data)):
        idx1 = row["idx1"]
        idx2 = row["idx2"]
        text1 = index_data.loc[idx1]["func"]
        text2 = index_data.loc[idx2]["func"]
        cosine_label = row["cosine_label"]
        input_examples.append(InputExample(texts=[text1, text2], label=cosine_label))
    return input_examples

def load_and_cache_examples(args: argparse, eval: bool = False, test: bool = False, max_examples_size: int = None) -> list:
    file_path = args.test_data_file if test else (args.valid_data_file if eval else args.train_data_file)
    pair_data = pd.read_csv(file_path, sep="\t")
    index_data = pd.read_json(args.index_data_file, lines=True, orient="records", encoding="utf-8").set_index("idx")
    return convert_idx_to_input_example(args, index_data, pair_data, max_examples_size)


def create_pooler_name(args: argparse) -> str:
    pooler_names = []
    if args.pooling_mode_cls:
        pooler_names.append("cls")
    if args.pooling_mode_max:
        pooler_names.append("max")
    if args.pooling_mode_mean:
        pooler_names.append("mean")
    return "-".join(pooler_names)


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

    parser.add_argument("--train_max_examples_size", type=int)
    parser.add_argument("--valid_max_examples_size", type=int)
    parser.add_argument("--test_max_examples_size", type=int)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--evaluate_during_training", action="store_true")

    parser.add_argument("--pooling_mode_mean", action="store_true")
    parser.add_argument("--pooling_mode_max", action="store_true")
    parser.add_argument("--pooling_mode_cls", action="store_true")
    
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--epochs_num", type=int, default=100)
    parser.add_argument("--evaluate_step", type=int, default=100)

    args = parser.parse_args()


    ## Create model output directory
    if args.model_name_or_path == "microsoft/codebert-base":
        pooler_name = create_pooler_name(args=args)
        model_save_path = os.path.join(args.output_dir, 
                                        args.model_name_or_path.replace("/", "-") + "_" + \
                                        pooler_name + "_" + \
                                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        ## Load pre-trained model
        transformer_model = models.Transformer(args.model_name_or_path)
        pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(),
                                        pooling_mode_mean_tokens=args.pooling_mode_mean,
                                        pooling_mode_cls_token=args.pooling_mode_cls,
                                        pooling_mode_max_tokens=args.pooling_mode_max)
        model = SentenceTransformer(modules=[transformer_model, pooling_model])
    else:
        model_save_path = args.model_name_or_path
        if not os.path.exists(model_save_path):
            print(f"Model path not found: {model_save_path}")
            print("Please check the model path.")
            exit()
        ## Load fine-tuned model
        model = SentenceTransformer(args.model_name_or_path)

    ## Create logger
    partition = "test" if args.do_test else ("valid" if args.do_eval else "train")
    datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = partition + "_log_" + datetime_now
    logging.basicConfig(format="%(asctime)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO, 
                        filename=os.path.join(model_save_path, f"{log_filename}.txt"))
    logger = logging.getLogger(__name__)


    if args.do_train:
        ## Create data loader and loss function for ContrastiveLoss
        train_dataset = load_and_cache_examples(args, eval=False, test=False, max_examples_size=args.train_max_examples_size)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
        train_loss = losses.CosineSimilarityLoss(model=model)

        ## Create data loader and evaluator for EmbeddingSimilarityEvaluator
        if args.evaluate_during_training:
            valid_dataset = load_and_cache_examples(args, eval=True, test=False, max_examples_size=args.valid_max_examples_size)
            embedding_sim_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(examples=valid_dataset,
                                                                                                name="valid",
                                                                                                batch_size=args.train_batch_size,
                                                                                                show_progress_bar=True)

        ## Train the model
        logger.info("********** Running training **********")
        logger.info("   Num examples = {}".format(len(train_dataset)))
        logger.info("   Num Epochs = {}".format(args.epochs_num))
        logger.info("   Batch size = {}".format(args.train_batch_size))
        logger.info("   Pooler name = {}".format(pooler_name))
        logger.info("   Evaluate step = {}".format(args.evaluate_step))
        logger.info("   Model save path: {}".format(model_save_path))

        if args.evaluate_during_training:
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                        epochs=args.epochs_num,
                        evaluator=embedding_sim_evaluator,
                        evaluation_steps=args.evaluate_step,
                        output_path=model_save_path,
                        save_best_model=True,
                        show_progress_bar=True)
        else:
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                        epochs=args.epochs_num,
                        output_path=model_save_path,
                        save_best_model=True,
                        show_progress_bar=True)

    if args.do_eval:
        ## Create data loader and evaluator for EmbeddingSimilarityEvaluator
        valid_dataset = load_and_cache_examples(args, eval=True, test=False, max_examples_size=args.valid_max_examples_size)
        embedding_sim_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(examples=valid_dataset,
                                                                                            name="eval",
                                                                                            batch_size=args.train_batch_size,
                                                                                            show_progress_bar=True)
        ## Evaluate the model
        logger.info("********** Running evaluation **********")
        logger.info("   Num examples = {}".format(len(valid_dataset)))
        logger.info("   Batch size = {}".format(args.train_batch_size))
        logger.info("   Pooler name = {}".format(pooler_name))
        logger.info("   Model save path: {}".format(model_save_path))

        model.evaluate(evaluator=embedding_sim_evaluator,
                        output_path=model_save_path,
                        show_progress_bar=True)
        
    if args.do_test:
        ## Load test data and index data
        pair_data = pd.read_csv(args.test_data_file, sep="\t")
        index_data = pd.read_json(args.index_data_file, lines=True, orient="records", encoding="utf-8").set_index("idx")

        ## Extract max_examples_size examples
        max_examples_size = args.test_max_examples_size
        if max_examples_size is not None:
            positive_pair = pair_data[pair_data["label"] == 1]
            negative_pair = pair_data[pair_data["label"] == 0]
            positive_pair = positive_pair.sample(n=max_examples_size//2, random_state=42)
            negative_pair = negative_pair.sample(n=max_examples_size//2, random_state=42)
            pair_data = pd.concat([positive_pair, negative_pair], ignore_index=True)

        ## Evaluate the model
        logger.info("********** Running test **********")
        logger.info("   Num examples = {}".format(len(pair_data)))
        logger.info("   Model save path: {}".format(model_save_path))

        ## Encode and calculate cosine simillarity
        encode_result = []
        for _, row in tqdm(pair_data.iterrows(), total=len(pair_data)):
            idx1 = row["idx1"]
            idx2 = row["idx2"]
            text1 = index_data.loc[idx1]["func"]
            text2 = index_data.loc[idx2]["func"]

            label = row["label"]
            edit_script = row["edit_script"]
            cosine_label = row["cosine_label"]

            ## Embedding & calculate cosine simillarity
            text1_embedding = model.encode(text1, convert_to_tensor=True)
            text2_embedding = model.encode(text2, convert_to_tensor=True)
            cosine_score = util.cos_sim(text1_embedding, text2_embedding).cpu().numpy().item()

            encode_result.append([text1, text2, label, edit_script, cosine_label, cosine_score])
        result = pd.DataFrame(encode_result, columns=["text1", "text2", "label", "edit_script", "cosine_label", "cosine_score"])
        filename = os.path.join(model_save_path, "{}.tsv".format(partition + "-res_" + datetime_now))
        result.to_csv(filename, sep="\t")
        

if __name__ == "__main__":
    main()