from tree_sitter import Language, Parser

def _get_all_nodes(tree: Parser) -> list:
    def _search_node(node):
        nonlocal nodes
        for child in node.children:
            nodes.append(child)
            _search_node(child)

    nodes = []
    nodes.append(tree.root_node)
    _search_node(tree.root_node)
    return nodes

def get_functions(tree: Parser) -> list:
    functions = []
    all_nodes = _get_all_nodes(tree)
    for node in all_nodes:
        if node.type == "function_definition":
            functions.append(node)
    return functions