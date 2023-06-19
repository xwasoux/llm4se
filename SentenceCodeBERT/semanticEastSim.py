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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--epochs_num', type=int, default=100)
    parser.add_argument('--evaluate_step', type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    model_save_path = 'output/training_codesearchnet_east_' + \
                        args.model_name_or_path.replace("/", "-")+'-' + \
                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    word_embedding_model = models.Transformer(args.model_name_or_path)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=False,
                                    pooling_mode_cls_token=True,
                                    pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    logging.info("Load Dataset from Pickle.")
    with open("data/semantic_east_sim/semantic_east_sim_dataset.pickle", "rb") as p:
        datasets = pickle.load(p)

    train_data = datasets[:int(len(datasets)*0.75)]
    valid_data = datasets[int(len(datasets)*0.75):]
    flatten_train_data = []
    for line in train_data:
        for each_data in line:
            flatten_train_data.append(each_data)
    flatten_valid_data = []
    for line in valid_data:
        for each_data in line:
            flatten_valid_data.append(each_data)
    logging.info(f'[Dataset size]\n\ttrain data:\n\t\ttype - {len(train_data)}\n\t\tlength - {len(flatten_train_data)}\n\tvalid data:\n\t\ttype - {len(valid_data)}\n\t\tlength - {len(flatten_valid_data)}')

    train_dataloader = DataLoader(flatten_train_data, shuffle=True, batch_size=args.train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    valid_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(flatten_valid_data,
                                                                        batch_size=args.train_batch_size,
                                                                        name='ses-valid')

    warmup_steps = math.ceil(len(train_dataloader) * args.epochs_num * 0.1)
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    logging.info("Start training...")

    model.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=valid_evaluator,
                epochs=args.epochs_num,
                evaluation_steps=args.evaluate_step,
                warmup_steps=warmup_steps,
                output_path=model_save_path
                )

logging.info("Finished to training model!")

# ##############################################################################
# #
# # Load the stored model and evaluate its performance on STS benchmark dataset
# #
# ##############################################################################

# test_samples = []
# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         if row['split'] == 'test':
#             score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
#             test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

# model = SentenceTransformer(model_save_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
# test_evaluator(model, output_path=model_save_path)


if __name__ == '__main__':
    main()
