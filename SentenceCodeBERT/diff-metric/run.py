import os
import sys
import csv
import glob
import gzip
import math
import random
import logging
import argparse
import pickle
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

def create_pooler_name(args:argparse) -> str:
    pooler_names = []
    if args.pooling_mode_cls:
        pooler_names.append("cls")
    if args.pooling_mode_max:
        pooler_names.append("max")
    if args.pooling_mode_mean:
        pooler_names.append("mean")
    return "-".join(pooler_names)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
    parser.add_argument('--base_model_save_path', type=str, default='output/training')

    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_evaluate', action="store_true")

    parser.add_argument('--pooling_mode_cls', action="store_true")
    parser.add_argument('--pooling_mode_max', action="store_true")
    parser.add_argument('--pooling_mode_mean', action="store_true")
    
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--epochs_num', type=int, default=100)
    parser.add_argument('--evaluate_step', type=int, default=100)

    parser.add_argument('--input_base_dir', type=str)
    parser.add_argument('--language', type=str)
    parser.add_argument('--train_datas', nargs='*')
    parser.add_argument('--upper_data_size', type=int)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
                        # handlers=[LoggingHandler()])

    pooler_name = create_pooler_name(args=args)
    
    model_save_path = os.path.join(args.base_model_save_path, 
                                    args.model_name_or_path.replace("/", "-") + '_' + \
                                        pooler_name + '_' + \
                                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                    )

    if args.do_train:

        ## model settings
        word_embedding_model = models.Transformer(args.model_name_or_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                        pooling_mode_mean_tokens=args.pooling_mode_mean,
                                        pooling_mode_cls_token=args.pooling_mode_cls,
                                        pooling_mode_max_tokens=args.pooling_mode_max)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        logging.info("Load Training Dataset from Pickle...")
        partition_type = "train"
                   
        train_data = []
        for pruning_type in args.train_datas:
            train_data_path = os.path.join(args.input_base_dir, args.language, partition_type, f"{pruning_type}.pickle")

            with open(train_data_path, "rb") as p:
                each_train_data = pickle.load(p)
                
            if args.upper_data_size:
                each_train_data = random.sample(each_train_data, args.upper_data_size)

            train_data.extend(each_train_data)

        logging.info(f'[Train Dataset size]\n\ttrain data:\n\t\tlength - {len(train_data)}')
        # logging.info(f'[Valid Dataset size]\n\tvalid data:\n\t\tlength - {len(valid_data)}')

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
        train_loss = losses.CosineSimilarityLoss(model=model)
        # valid_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(valid_data,
        #                                                                     batch_size=args.train_batch_size,
        #                                                                     name='ses-valid')

        warmup_steps = math.ceil(len(train_dataloader) * args.epochs_num * 0.1)
        logging.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        logging.info("Start training...")
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                    # evaluator=valid_evaluator,
                    epochs=args.epochs_num,
                    evaluation_steps=args.evaluate_step,
                    warmup_steps=warmup_steps,
                    output_path=model_save_path
                    )

        logging.info("Finished to training model!")

    # if args.do_evaluate:
    #     test_samples = []
    #     with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    #         reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    #         for row in reader:
    #             if row['split'] == 'test':
    #                 score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
    #                 test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    #     model_tuned = SentenceTransformer(model_save_path)
    #     test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=model_tuned, name='sts-test')
    #     test_evaluator(model_tuned, output_path=model_save_path)


if __name__ == '__main__':
    main()

