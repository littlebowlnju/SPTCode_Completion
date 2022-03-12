import tree_sitter
from tree_sitter import Language, Parser
import re

LANGUAGE = Language('data/ast/build/my-languages.so', 'java')

parser = Parser()

SOURCE_PREFIX_POSTFIX = ['class A{ ', ' }']

PATTERNS_METHOD_ROOT = """
    (program
        (class_declaration
            body: (class_body
                (method_declaration) @method_root)
        )
    )
    """

PATTERNS_METHOD_BODY = """
    (method_declaration
        body: (block) @body
    )
    """

PATTERNS_METHOD_NAME = """
    (method_declaration
        name: (identifier) @method_name
    )
    """

PATTERNS_METHOD_INVOCATION = """
    (method_invocation
        name: (identifier) @method_invocation
    )
    """

STATEMENT_ENDING_STRINGS = ['statement', 'expression', 'declaration']


def camel_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def split_identifier(identifier):
    """
    Split identifier into a list of subtokens.
    Tokens except characters and digits will be eliminated.
    :param identifier: given identifier
    :return: list[str], list of subtokens
    """
    words = []

    word = re.sub(r'[^a-zA-Z0-9]', ' ', identifier)
    word = re.sub(r'(\d+)', r' \1 ', word)
    split_words = word.strip().split()
    for split_word in split_words:
        camel_words = camel_split(split_word)
        for camel_word in camel_words:
            words.append(camel_word.lower())

    return words


def parse_ast(source):
    """
    Parse the given code into corresponding ast.
    :param source: str, code in string
    :return: tree_sitter.Node: Method/Function root node
    """
    parser.set_language(LANGUAGE)
    source = SOURCE_PREFIX_POSTFIX[0] + source + SOURCE_PREFIX_POSTFIX[1]
    tree = parser.parse(source.encode('utf-8').decode('unicode_escape').encode())
    root = tree.root_node
    query = LANGUAGE.query(PATTERNS_METHOD_ROOT)
    captures = query.captures(root)
    root = captures[0][0]
    return root


def is_statement_node(node):
    """
    Return whether the node id a statement level node.
    :param node: tree_sitter.Node, Node to be queried
    :return: True if given node is a statement node
    """
    endings = STATEMENT_ENDING_STRINGS
    end = node.type.split('_')[-1]
    if end in endings:
        return True
    else:
        return False


def get_node_type(node):
    """
    Return the type of node
    :param node: tree_sitter.Node, Node to be queried
    :return: str, type of the node
    """
    return f'{node.type}_statement'


def __statement_xsbt(node):
    """
    Method used to generate X-SBT recursively.
    :param node: tree_sitter.Node, root node to traversal
    :return: list[str], List of strings representing node types
    """
    xsbt = []
    if len(node.children) == 0:
        if is_statement_node(node):
            xsbt.append(get_node_type(node))
    else:
        if is_statement_node(node):
            xsbt.append(f'{get_node_type(node)}__')
        len_before = len(xsbt)
        for child in node.children:
            xsbt += __statement_xsbt(child)
        if len_before == len(xsbt) and len_before != 0:
            xsbt[-1] = get_node_type(node)
        elif is_statement_node(node):
            xsbt.append(f'__{get_node_type(node)}')

    return xsbt


def generate_statement_xsbt(node):
    """
    Generate X-SBT string.
    :param node: tree_sitter.Node, root node to traversal
    :return: str, X-SBT string
    """
    query = LANGUAGE.query(PATTERNS_METHOD_BODY)
    captures = query.captures(node)
    node = captures[0][0]
    tokens = __statement_xsbt(node)
    return ''.join(tokens)


def get_node_name(source, node):
    """
    Get node name
    :param source: source code string
    :param node: tree_sitter.Node, node instance
    :return: str, name of node
    """
    if node.is_named:
        return source[node.start_byte - len(SOURCE_PREFIX_POSTFIX[0]):
                      node.end_byte - len(SOURCE_PREFIX_POSTFIX[0])]
    return ''


def get_method_name(source, root):
    """
    Return the name of method/function.
    :param source: Source code string
    :param root: tree_sitter.Node, Method/Function root node
    :return:
    """
    query = LANGUAGE.query(PATTERNS_METHOD_NAME)
    captures = query.captures(root)
    if len(captures) == 0:
        return ''
    return get_node_name(source, captures[0][0])


def extract_method_invocation(source, root):
    """
    Extract method invocation sequence from given root.
    :param source: str, source code string
    :param root: tree_sitter.Node, Node to be extracted from
    :return: list[str], list of method invocation strings
    """
    query = LANGUAGE.query(PATTERNS_METHOD_INVOCATION)
    captures = query.captures(root)
    return [get_node_name(source=source, node=capture[0]) for capture in captures]


def extract_nl_from_code(source, root, name=None, replace_method_name=False):
    """
    Extract nl tokens from given source code, including split name and method invocations.
    :param source: source code string
    :param root: tree_sitter.Node, root of code
    :param name: optional, name of method/function
    :param replace_method_name: Whether to replace method name and returns a version that without names additionally
    :return: Union[(str, str), str]:
            - Nl string
            - Nl string without method name
    """
    tokens = []
    tokens_wo_name = []

    if name is None:
        name = get_method_name(source=source, root=root)
    name_tokens = split_identifier(name)
    tokens += name_tokens

    invocations = extract_method_invocation(source=source, root=root)
    for invocation in invocations:
        subtokens = split_identifier(invocation)
        tokens += subtokens
        tokens_wo_name += subtokens

    if replace_method_name:
        return ' '.join(tokens), ' '.join(tokens_wo_name)
    else:
        return ' '.join(tokens)


def generate_single_ast_nl(source, name=None, replace_method_name=False):
    """
    Generate AST sequence and nl sequence for a single source code sample.
    :param source: Source code string
    :param name: optional, name of method/function
    :param replace_method_name: whether to replace method name and returns a version that without names additionally
    :return: Union[(str, str), (str, str, str)]:
            - AST sequence in string
            - NL sequence in string
    """
    root = parse_ast(source=source)
    ast = generate_statement_xsbt(node=root)
    if replace_method_name:
        nl, nl_wo_name = extract_nl_from_code(source=source,
                                              root=root,
                                              name=name,
                                              replace_method_name=replace_method_name)
        return ast, nl, nl_wo_name
    else:
        nl = extract_nl_from_code(source=source, root=root, name=name)
        return ast, nl
