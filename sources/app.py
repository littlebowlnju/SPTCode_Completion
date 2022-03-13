import argparse
from flask import Flask, jsonify, request
from data.data_utils import tokenize_source, parse_for_completion, generate_input
from data.ast.ast_parser import generate_single_ast_nl
from args import add_args
from completion import run_completion
import os
import time

app = Flask(__name__)

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

SOURCE2 = "public void map(Text key, LongWritable value, OutputCollector<Text, Text> output,Reporter reporter) throws IOException {" \
          "String name = key.toString();" \
          "long longValue = value.get();" \
          """reporter.setStatus("starting " + name + " ::host = " + hostName);""" \
          "PRED " \
          "parseLogFile(fs, new Path(name), longValue, output, reporter);" \
          "long tEnd = System.currentTimeMillis();" \
          "long execTime = tEnd - tStart;" \
          """reporter.setStatus("finished " + name + " ::host = " + hostName + " in " + execTime / 1000 + " sec.");}"""

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

@app.route("/", methods=['POST', 'GET'])
def generate_result():
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

    if request.method == 'POST':
        data = request.get_json()
        predictions, prediction_scores = run_completion(
            args=args,
            source=data,
            trained_model=args.trained_model,
            trained_vocab=args.trained_vocab)
        result = [
            {"predictions" : str(predictions)},
            {"prediction_scores" : str(prediction_scores)}
        ]
        return jsonify(result)
    else:
        return jsonify({"about": "Hello World"})


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

    app.run()

