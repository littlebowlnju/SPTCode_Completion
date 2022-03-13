import argparse
from data.data_utils import tokenize_source, parse_for_completion, generate_input
from data.ast.ast_parser import generate_single_ast_nl
from args import add_args
from completion import run_completion
import os
import time

# ------ original source ------------
# SOURCE = "public int[] twoSum(int[] nums, int target) {" \
#          "int n = nums.length;" \
#          "for (int i = 0; i < n; ++i) {" \
#          "for (int j = i + 1; j < n; ++j) {" \
#          "if (nums[i] + nums[j] == target) {" \
#          "return new int[]{i, j};}}}" \
#          "return new int[0];}"

# ------- source with PRED token -------------
SOURCE = "public int[] twoSum(int[] nums, int target) {" \
         "PRED " \
         "for (int i = 0; i < n; ++i) {" \
         "for (int j = i + 1; j < n; ++j) {" \
         "if (nums[i] + nums[j] == target) {" \
         "return new int[]{i, j};}}}" \
         "return new int[0];}"


# test tokenize source code into tokens list
def tokenize_source_test():
    tokens = tokenize_source(SOURCE)
    print(tokens)


# test generate ast and nl from source code
def generate_ast_nl_test():
    ast, nl = generate_single_ast_nl(SOURCE)
    print(ast)
    print('-' * 100)
    print(nl)


# to see what dataset item like
# def dataset_item_test():
#     source_path = './dataset/source.txt'
#     target_path = './dataset/target.txt'
#     codes, ast, names, target = parse_for_completion(source_path, target_path)
#     print('-'*100)
#     print(codes)
#     print('-'*100)
#     print(ast)
#     print('-'*100)
#     print(names)
#     print('-'*100)
#     print(target)


if __name__ == '__main__':
    # tokenize_source_test()

    # generate_ast_nl_test()

    # dataset_item_test()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    add_args(parser)
    args = parser.parse_args()


    args.output_root = os.path.join(
        '..',
        'outputs',
        '{}_{}'.format(args.model_name, time.strftime('%Y%m%d_%H%M%S', time.localtime())))
    args.checkpoint_root = os.path.join(args.output_root, 'checkpoints')
    args.tensor_board_root = os.path.join(args.output_root, 'runs')

    predictions, prediction_scores = run_completion(
        args=args,
        source=SOURCE,
        trained_model=args.trained_model,
        trained_vocab=args.trained_vocab)