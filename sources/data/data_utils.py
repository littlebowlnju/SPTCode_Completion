from antlr4 import InputStream
import re
import os

from data.antlr_parser.java.Java8Lexer import Java8Lexer
from data.ast.ast_parser import generate_single_ast_nl
from data.vocab import load_vocab, Vocab

from data.data_collator import get_concat_batch_inputs, get_batch_inputs

code_vocab = load_vocab('../pre_trained/vocabs/', 'code')
nl_vocab = load_vocab('../pre_trained/vocabs/', 'nl')
ast_vocab = load_vocab('../pre_trained/vocabs/', 'ast')

STRING_MATCHING_PATTERN = re.compile(r'([bruf]*)(\"\"\"|\'\'\'|\"|\')(?:(?!\2)(?:\\.|[^\\]))*\2')


def replace_string_literal(source):
    """
    Replace the string literal in source code with ``<STR>``.
    :param source: Source code in string
    :return: str: Code after replaced
    """
    return re.sub(pattern=STRING_MATCHING_PATTERN, repl='___STR', string=source)


def trim_spaces(string):
    """
    Replace consecutive spaces with a single whitespace.
    :param string: String
    :return: str: Replaced string
    """
    return re.sub(r'\s+', ' ', string).strip()


def tokenize_source(source, use_regular=False):
    """
    Tokenize the source code into tokens.
    :param source: Source code in string
    :param use_regular: Whether to use regular tokenize method, default to False
    :return: str: Tokenized code, delimited by whitespace, string literal will be replaced by ``___STR``
    """
    # if use_regular:
    #     code = replace_string_literal()

    input_stream = InputStream(source)
    lexer = Java8Lexer(input_stream)
    tokens = [token.text for token in lexer.getAllTokens()]
    code = replace_string_literal(' '.join(tokens))
    return trim_spaces(code)


def restore_source(sub_source):
    """
        Transfer split source to source code, which can be parsed into AST.

        Args:
            sub_source (str): Split code

        Returns:
            str: Source code that can be parsed

    """
    tokens = sub_source.split()
    is_subtoken = False
    restored_source = ''
    for token in tokens:
        if token == '_':
            is_subtoken = True
            continue
        if token == 'PRED':
            token = Vocab.MSK_TOKEN
        if is_subtoken:
            restored_source += token.capitalize()
        else:
            restored_source += f' {token}'
        is_subtoken = False
    return restored_source.strip()


def generate_input(args, source):
    """
    Generate input from source code sting
    :param source: str, source code string
    :return: code, ast, nl of source code
    """
    # split source code to split source
    code = tokenize_source(source).strip()
    print('-------split code---------')
    print(code)
    # restore
    code = restore_source(code)
    print('-------restored code--------')
    print(code)
    # generate ast and nl from restored code
    ast, name = generate_single_ast_nl(code)
    print(ast)
    print(name)

    # transfer to sequence
    code = [code]
    ast = [ast]
    name = [name]

    model_inputs = {}
    model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
        code_raw=code,
        code_vocab=code_vocab,
        max_code_len=args.max_code_len,
        ast_raw=ast,
        ast_vocab=ast_vocab,
        max_ast_len=args.max_ast_len,
        nl_raw=name,
        nl_vocab=nl_vocab,
        max_nl_len=args.max_nl_len,
        no_ast=args.no_ast,
        no_nl=args.no_nl
    )
    return model_inputs

def load_lines(path):
    """
    Load lines from given path.

    Args:
        path (str): Dataset file path

    Returns:
        list: List of lines

    """
    with open(path, encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return lines


def parse_for_completion(source_path, target_path):
    """
    Load and parse for code completion.

    Args:
        source_path (str): Path of source
        target_path (str): Path of target

    Returns:
        (list[str], list[str], list[str], list[str]):
            - List of strings: source code
            - List of strings: AST sequence
            - List of strings: name sequence
            - List of strings: target code

    """
    source_lines = load_lines(source_path)
    target_lines = load_lines(target_path)
    assert len(source_lines) == len(target_lines)

    codes = []
    asts = []
    names = []
    targets = []
    for source, target in zip(source_lines, target_lines):
        try:
            source = restore_source(source)
            target = restore_source(target)
            ast, name = generate_single_ast_nl(source=source)
            codes.append(source)
            asts.append(ast)
            names.append(name)
            targets.append(target)
        except Exception:
            continue
    return codes, asts, names, targets


def my_parse_for_completion(args, source_path, target_path):
    code, ast, name, targets = parse_for_completion(source_path, target_path)
    model_inputs = {}
    model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
        code_raw=code,
        code_vocab=code_vocab,
        max_code_len=args.max_code_len,
        ast_raw=ast,
        ast_vocab=ast_vocab,
        max_ast_len=args.max_ast_len,
        nl_raw=name,
        nl_vocab=nl_vocab,
        max_nl_len=args.max_nl_len,
        no_ast=args.no_ast,
        no_nl=args.no_nl
    )
    model_inputs['labels'], _ = get_batch_inputs(batch=targets,
                                                 vocab=code_vocab,
                                                 processor=Vocab.eos_processor,
                                                 max_len=args.completion_max_len)
    return model_inputs