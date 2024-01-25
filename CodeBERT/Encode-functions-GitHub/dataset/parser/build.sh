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