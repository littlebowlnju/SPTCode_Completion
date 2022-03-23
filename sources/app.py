import argparse
import uuid
import redis
from threading import Thread
import enums
from flask import Flask, jsonify, request
from data.data_utils import tokenize_source, parse_for_completion, generate_input
from data.ast.ast_parser import generate_single_ast_nl
from args import add_args
from typing import Union, Tuple
from transformers import BartConfig, Seq2SeqTrainingArguments, EarlyStoppingCallback, \
    IntervalStrategy, SchedulerType
from data.vocab import Vocab, load_vocab, init_vocab
from model.bart import BartForClassificationAndGeneration
from utils.general import count_params, human_format, layer_wise_parameters
import os
import time
import json

app = Flask(__name__)
pool = redis.ConnectionPool(host='localhost', port=6379, max_connections=50)
redis_ = redis.Redis(connection_pool=pool, decode_responses=True)
db_key_query = 'query'
db_key_result = 'result'
batch_size = 1
MODEL_MODE_GEN = 'bart_gen'
args = None

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

add_args(parser)
# args = parser.parse_args()
args, unknown = parser.parse_known_args()

args.output_root = os.path.join(
    '..',
    'outputs',
    '{}_{}'.format(args.model_name, time.strftime('%Y%m%d_%H%M%S', time.localtime())))
args.checkpoint_root = os.path.join(args.output_root, 'checkpoints')
args.tensor_board_root = os.path.join(args.output_root, 'runs')

code_vocab = load_vocab(vocab_root=args.trained_vocab, name=args.code_vocab_name)

config = BartConfig.from_json_file(os.path.join(args.trained_model, 'config.json'))
model = BartForClassificationAndGeneration.from_pretrained(os.path.join(args.trained_model, 'pytorch_model.bin'),
                                                           config=config)
model.set_model_mode(enums.MODEL_MODE_GEN)

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
    from completion import complete
    if request.method == 'POST':
        data = request.get_data().decode("utf-8")
        print(data)
        #print(args)
        predictions, prediction_scores = complete(
            args=args,
            source=data,
            model=model,
            code_vocab=code_vocab
        )
        pre_scores = []
        for score in prediction_scores:
            pre_scores.append("%.2f%%" % (score * 100))
        result = {
            "predictions": predictions,
            "prediction_scores": pre_scores
        }
        return json.dumps(result)
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

    # app.run(host="172.29.7.224", port=1818, debug=True)
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)

    # t = Thread(target=print_test, args=(batch_size,))
    # t.start()
    app.run()
