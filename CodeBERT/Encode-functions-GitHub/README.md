# Encode functions GitHub

## Task Definition

Given a code snippet, the task is to encode the code into a fixed-size vector. 

## Updates

## Data

The dataset we use is some public GitHub repositories.

### Preprocess

```bash
cd dataset/parser
parser_dir=$(pwd)

mkdir -p tree-sitter
cd tree-sitter

echo "Cloning tree-sitter repos..."
git clone https://github.com/tree-sitter/tree-sitter-c
git clone https://github.com/tree-sitter/tree-sitter-cpp
git clone https://github.com/tree-sitter/tree-sitter-typescript
git clone https://github.com/tree-sitter/tree-sitter-go
git clone https://github.com/tree-sitter/tree-sitter-javascript
git clone https://github.com/tree-sitter/tree-sitter-python
git clone https://github.com/tree-sitter/tree-sitter-ruby
git clone https://github.com/tree-sitter/tree-sitter-php
git clone https://github.com/tree-sitter/tree-sitter-java
git clone https://github.com/tree-sitter/tree-sitter-c-sharp

cd $parser_dir
python build.py
```


```bash
cd dataset
python preprocess.py \
    --girhub_repo https://github.com/zephyrproject-rtos/zephyr.git \
    --branch main \
    --language c \

```



### Embedding

```bash
cd code
python run.py \
    --model_name_or_path microsoft/codebert-base \
    --data_path ../dataset/inputs/zephyrproject-rtos/zephyr/inputs.jsonl \
    --dim 2 \
    --output_path ./outputs
```
