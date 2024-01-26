import os
from git import Repo
from tree_sitter import Language, Parser

tree_sitter_dir = os.path.join(os.path.dirname(__file__), 'tree-sitter')
if not os.path.exists(tree_sitter_dir):
    os.makedirs(tree_sitter_dir)

# Download the tree-sitter-java grammar
repo = Repo.clone_from(url="https://github.com/tree-sitter/tree-sitter-java", 
                        to_path=os.path.join(tree_sitter_dir, 'tree-sitter-java'))

Language.build_library(
    # Store the library in the `build` directory
    'my-languages.so',

    # Include one or more languages
    [
        os.path.join("tree-sitter", 'tree-sitter-java'),
    ]
)