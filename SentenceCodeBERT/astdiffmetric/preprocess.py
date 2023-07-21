import json
import sys
import glob
import pickle
import pandas as pd
import logging
import argparse
from pprint import pprint

from sentence_transformers import InputExample

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def getPaths(pathName:str) -> list:
    condition = f'{pathName}/*/*.jsonl'
    return glob.glob(condition, recursive=True)

def charLenCond(jsonlPaths:list, upperSize:int) -> list:
    upperCharLenCode = []

    for path in jsonlPaths:
        with open(path) as f:
            try:
                oneLine = json.loads(f.readline())
            except:
                continue
            
        sourceSizeChar = oneLine["sourceSizeChar"]
        if sourceSizeChar <= upperSize:
            upperCharLenCode.append(path)

    return upperCharLenCode

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str)
    parser.add_argument("--valid_data", type=str)
    parser.add_argument("--test_data", type=str)

    parser.add_argument("--upper_char_size", type=int)

    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    allDataPath = [args.train_data, args.valid_data, args.test_data]

    for dataPath in allDataPath:
        allJsonl = getPaths(dataPath)
        logging.info(len(allJsonl))

        upperCharLenPaths = charLenCond(jsonlPaths=allJsonl, upperSize=args.upper_char_size)
        logging.info(len(upperCharLenPaths))
        
        allInputExamples = []
        for path in upperCharLenPaths:
            with open(path) as f:
                jsonlData = [json.loads(l) for l in f.readlines()]
            
            for line in jsonlData:
                allInputExamples.append(InputExample(guid=line["index"], 
                                                     texts=[line["source"], line["target"]], 
                                                     label=line["cosSimNodeDel"]))

        logging.info(f"InputExample Data -> {len(allInputExamples)}")

        storeFileName = f"{args.output_dir}/{dataPath.split('/')[-1]}.pickle"
        with open(storeFileName, "wb") as p:
            pickle.dump(allInputExamples, p)
        logging.info(f"Stored pickle data -> {storeFileName}")
    
    return None



if __name__ == '__main__':
    main()

