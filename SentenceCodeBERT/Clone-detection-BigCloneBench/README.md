# Clone-detection-BigCloneBench

## Task Definition

Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by F1 score.

## Updates


## Dataset

The dataset we use is [BigCloneBench](https://www.cs.usask.ca/faculty/croy/papers/2014/SvajlenkoICSME2014BigERA.pdf) and filtered following the paper [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree](https://arxiv.org/pdf/2002.08653.pdf).

### Data Format

1. dataset/data.jsonl is stored in jsonlines format. Each line in the uncompressed file represents one function.  One row is illustrated below.

   - **func:** the function

   - **idx:** index of the example

2. train.txt/valid.txt/test.txt provide examples, stored in the following format:    idx1	idx2	label

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  901,028  |
| Dev   |  415,416  |
| Test  |  415,416  |

## Fine-tune

```bash
cd code
python3 run.py \
    --model_name_or_path microsoft/codebert-base \
    --index_data_file ../dataset/data.jsonl \
    --train_data_file ../dataset/train.tsv \
    --valid_data_file ../dataset/valid.tsv \
    --output_dir ./saved_model \
    --do_train \
    --train_batch_size 32 \
    --epochs_num 10 \
    --evaluate_step 100 \
    --train_max_examples_size 16000 \
    --valid_max_examples_size 2000 \
    --pooling_mode_mean
```

## Repository Clustering

```bash
cd repo-clustering/data/parser
python3 build.py

cd ../../
python3 repo-clustering.py \
    --model_name_or_path ../code/saved_model/microsoft-codebert-base_xxxx_yyyy-mm-dd_hh-mm-ss \
    --remote_repo_url https://github.com/FasterXML/jackson-databind.git \
    --output_dir ./output

```