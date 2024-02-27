import os
import logging
from tree_sitter import Language, Parser

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO
                    )

logging.info("Building parsers...")
Language.build_library(
  # Store the library in the `build` directory
  'my-languages.so',

  # Include one or more languages
  [
    os.path.join("tree-sitter", 'tree-sitter-go'),
    os.path.join("tree-sitter", 'tree-sitter-javascript'),
    os.path.join("tree-sitter", 'tree-sitter-python'),
    # os.path.join("tree-sitter", 'tree-sitter-php'),
    os.path.join("tree-sitter", 'tree-sitter-java'),
    os.path.join("tree-sitter", 'tree-sitter-c-sharp'),
    os.path.join("tree-sitter", 'tree-sitter-cpp'),
    os.path.join("tree-sitter", 'tree-sitter-c'),
  ]
)
logging.info("Done building parsers...")