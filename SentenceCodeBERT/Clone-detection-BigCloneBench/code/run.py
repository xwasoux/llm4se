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

def convert_idx_to_input_example(index_data: pd.DataFrame, pair_data: pd.DataFrame) -> list:
    input_examples = []
    for _, row in tqdm(index_data.iterrows()):
        idx1 = row['idx1']
        idx2 = row['idx2']
        label = row['label']
        text1 = pair_data.loc[idx1]['func']
        text2 = pair_data.loc[idx2]['func']
        input_examples.append(InputExample(texts=[text1, text2], label=label))
    return input_examples

def load_and_cache_examples(args: argparse, eval: bool = False, test: bool = False) -> list:
    file_path = args.test_data_file if test else (args.valid_data_file if eval else args.train_data_file)
    index_data = pd.read_csv(file_path, sep='\t', header=None, names=['idx1', 'idx2', 'label'])
    pair_data = pd.read_json(args.index_data_file, lines=True, orient='records', encoding='utf-8')
    pair_data = pair_data.set_index('idx')
    return convert_idx_to_input_example(index_data, pair_data)


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
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base', required=True)
    parser.add_argument('--train_data_file', type=str, default=None, required=True)
    parser.add_argument('--index_data_file', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default='./saved_models', required=True)

    ## Other parameters
    parser.add_argument('--valid_data_file', type=str, default=None)
    parser.add_argument('--test_data_file', type=str, default=None)

    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_evaluate', action="store_true")
    parser.add_argument('--do_test', action="store_true")

    parser.add_argument('--pooling_mode_mean', action="store_true")
    parser.add_argument('--pooling_mode_max', action="store_true")
    parser.add_argument('--pooling_mode_cls', action="store_true")
    
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--epochs_num', type=int, default=100)
    parser.add_argument('--evaluate_step', type=int, default=100)

    args = parser.parse_args()


    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO, 
                        filename=os.path.join(args.output_dir, 'log.txt'))

    ## Create model output directory
    pooler_name = create_pooler_name(args=args)
    model_save_path = os.path.join(args.output_dir, 
                                    args.model_name_or_path.replace("/", "-") + '_' + \
                                    pooler_name + '_' + \
                                    datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if args.model_name_or_path:
        word_embedding_model = models.Transformer(args.model_name_or_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                        pooling_mode_mean_tokens=args.pooling_mode_mean,
                                        pooling_mode_cls_token=args.pooling_mode_cls,
                                        pooling_mode_max_tokens=args.pooling_mode_max)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    if args.do_train:
        ## Create data loader and loss function for ContrastiveLoss
        train_dataset = load_and_cache_examples(args, eval=False, test=False)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
        train_loss = losses.ContrastiveLoss(model=model)

        ## Create data loader and evaluator for BinaryClassificationEvaluator
        valid_dataset = load_and_cache_examples(args, eval=True, test=False)
        binary_acc_evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(examples=valid_dataset,
                                                                                            name='valid',
                                                                                            batch_size=args.train_batch_size,
                                                                                            write_csv=True,
                                                                                            show_progress_bar=True)

        ## Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                    epochs=args.epochs_num,
                    evaluator=binary_acc_evaluator,
                    evaluation_steps=args.evaluate_step,
                    output_path=model_save_path,
                    save_best_model=True,
                    show_progress_bar=True
                    )

if __name__ == '__main__':
    main()