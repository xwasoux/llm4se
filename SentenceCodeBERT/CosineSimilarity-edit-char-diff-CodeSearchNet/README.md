# Clone-detection-BigCloneBench

## Update

2024-02-02: Initial release.


## Task Definition

Given two codes as input, the task is to perform Contrastive Learning with real labels ranging from 0.0 to 1.0. The model is evaluated by Pearson's correlation coefficient.


## Dataset

The dataset we use is [BigCloneBench](https://www.cs.usask.ca/faculty/croy/papers/2014/SvajlenkoICSME2014BigERA.pdf) and filtered following the paper [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree](https://arxiv.org/pdf/2002.08653.pdf).

### Data Format

1. dataset/data.jsonl is stored in jsonlines format. Each line in the uncompressed file represents one function.  One row is illustrated below.

   - **func:** the function

   - **idx:** index of the example

2. ed-train.txt/ed-valid.txt/ed-test.txt provide examples, stored in the following format:    idx1	idx2	label(0/1)   edit_distance   cosine_label

### Data download & preprocess

Data download & example of preprocessing are shown below:

```bash
cd data
gdown 1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
rm Cleaned_CodeSearchNet.zip

python3 preprocess.py --language python
```

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  901,028  |
| Dev   |  415,416  |
| Test  |  415,416  |

## Fine-tune

An example of training is shown below:

```bash
cd code
python3 run.py \
    --model_name_or_path microsoft/codebert-base \
    --index_data_file ../dataset/python-ed-data.jsonl \
    --train_data_file ../dataset/python-ed-train.txt \
    --valid_data_file ../dataset/python-ed-valid.txt \
    --language python \
    --output_dir ./saved_model \
    --do_train \
    --evaluate_during_training \
    --train_batch_size 32 \
    --epochs_num 10 \
    --evaluate_step 100 \
    --train_max_examples_size 16000 \
    --valid_max_examples_size 2000 \
    --pooling_mode_mean
```

## Test & Evaluate

An example of testing is shown below:

```bash
cd code
python3 run.py \
    --model_name_or_path ./saved_model/microsoft-codebert-base_xxxx_yyyy-mm-dd_hh-mm-ss \
    --index_data_file ../dataset/data.jsonl \
    --test_data_file ../dataset/ed-test.txt \
    --do_test \
    --test_max_examples_size 1000
```

An example of evaluating is shown below:
```bash
cd pearson
python3 evaluator.py \
    --model_name_or_path ../code/saved_model/microsoft-codebert-base_xxxx_yyyy-mm-dd_hh-mm-ss \
    --test_result_file ../code/saved_model/microsoft-codebert-base_xxxx_yyyy-mm-dd_hh-mm-ss/test-predict_xxxx_yyyy-mm-dd_hh-mm-ss.txt
```
