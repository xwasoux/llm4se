import sys
import os
import gzip
import csv
import logging
from datetime import datetime
import argparse

from model import CodeBertEncoder
from analyser import EmbeddingAnalyser

def sep_csv(csv_data_path):
    code_list = []
    label_list = []

    with open(csv_data_path) as f:
        reader = csv.reader(f)
        reader_list = [row for row in reader]

    for row in reader_list:
        code_list.append(row[0])
        label_list.append(row[1])
    
    return code_list, label_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)

    parser.add_argument("--torch_transformer", action="store_true")
    parser.add_argument("--sentence_transformer", action="store_true")
    parser.add_argument("--dimention", type=int)

    parser.add_argument("--csv_data_path", type=str)
    parser.add_argument("--outputdir", type=str)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )

    code_list, label_list = sep_csv(args.csv_data_path)

    if args.torch_transformer:
        logging.info("torch_transformer")

        code_encoder = CodeBertEncoder()
        code_encoder.input(sentences=code_list)

        logging.info("Embedding Codes...")
        embeddings = code_encoder.embedding()

        ## Todo: separate class (named Analyser)
        logging.info(f"Reducing to {args.dimention} Dimension...")
        analyser = EmbeddingAnalyser(codes=code_list,
                            embeddings=embeddings, labels=label_list)
        analyser.reduce_dimension(dim=args.dimention)

        logging.info(f"Plotting Destributed Representation of Codes...")
        analyser.plot_embedding()

        logging.info(f"Calculating Cosine Similarity of Code pairs...")
        analyser.conineSimilarity(dir=args.outputdir)

    return None


if __name__ == "__main__":
    main()