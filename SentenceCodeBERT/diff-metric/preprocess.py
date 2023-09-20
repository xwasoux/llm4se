import os
import json
import sys
import glob
import pickle
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from sentence_transformers import InputExample

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def getDirPaths(pathName:str) -> list:
    condition = f'{pathName}/*/'
    return glob.glob(condition)

def getJsonlPaths(pathName:str) -> list:
    condition = f'{pathName}/*.jsonl'
    return glob.glob(condition)

def mkDir(pathStr:str) -> None:
    os.makedirs(pathStr, mode=0o777, exist_ok=True)
    return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    langTypes = getDirPaths(args.input_dir)

    for langPath in langTypes:
        lang = langPath.split("/")[-2]
        logging.info(f"=== {lang} ===")

        eachTypesPaths = getDirPaths(langPath)

        for purposePath in eachTypesPaths:
            purposeType = purposePath.split("/")[-2]
            logging.info(f"== {purposeType} ==")

            logging.info(purposePath)
            
            deletionTypePaths = getDirPaths(purposePath)
            logging.info(deletionTypePaths)

            for deleteTypePath in deletionTypePaths:
                deleteType = deleteTypePath.split("/")[-2]

                allDataPath = getJsonlPaths(deleteTypePath)

                allInputExamples = []

                for dataPath in tqdm(allDataPath):
                    with open(dataPath) as f:
                        jsonlData = [json.loads(l) for l in f.readlines()]

                    for line in jsonlData:
                        allInputExamples.append(InputExample(guid=f"index", 
                                                            texts=[line["originalCode"], line["editedCode"]], 
                                                            label=line["cosSimChar"]))

                logging.info(f"InputExample Data : {len(allInputExamples)}")

                storeDir = os.path.join(args.output_dir, lang, purposeType)
                os.makedirs(storeDir, mode=0o777, exist_ok=True)
                storeFileName = os.path.join(storeDir, f"{deleteType}.pickle")
                with open(storeFileName, "wb") as p:
                    pickle.dump(allInputExamples, p)
                logging.info(f"Stored pickle data -> {storeFileName}")


    return None


if __name__ == '__main__':
    main()

