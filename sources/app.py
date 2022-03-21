import argparse
import uuid
import redis
from threading import Thread
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

"""
def load_model(
        trained_model: Union[BartForClassificationAndGeneration, str],
        trained_vocab: Union[Tuple[Vocab, Vocab, Vocab], str]
):
    # -------------------------------
    # vocabs
    # -------------------------------
    global code_vocab
    global model
    if trained_vocab:
        if isinstance(trained_vocab, tuple):
            # app.info('Vocabularies are passed through parameter')
            assert len(trained_vocab) == 3
            code_vocab, ast_vocab, nl_vocab = trained_vocab
        else:
            # app.info('Loading vocabularies from files')
            code_vocab = load_vocab(vocab_root=trained_vocab, name='code')
            ast_vocab = load_vocab(vocab_root=trained_vocab, name='ast')
            nl_vocab = load_vocab(vocab_root=trained_vocab, name='nl')

    # app.info(f'The size of code vocabulary: {len(code_vocab)}')
    # app.info(f'The size of nl vocabulary: {len(nl_vocab)}')
    # app.info(f'The size of ast vocabulary: {len(ast_vocab)}')
    # app.info('Vocabularies built successfully')

    # ------------------------------
    # model
    # ------------------------------
    # app.info('-' * 100)
    if trained_model:
        if isinstance(trained_model, BartForClassificationAndGeneration):
            # app.info('Model is passed through parameter')
            model = trained_model
        else:
            # app.info('Loading the model from file')
            config = BartConfig.from_json_file(os.path.join(trained_model, 'config.json'))
            model = BartForClassificationAndGeneration.from_pretrained(os.path.join(trained_model, 'pytorch_model.bin'),
                                                                       config=config)
    model.set_model_mode(MODEL_MODE_GEN)
    # log model statistic
    # app.info('Trainable parameters: {}'.format(human_format(count_params(model))))
    table = layer_wise_parameters(model)
    # app.debug('Layer-wised trainable parameters:\n{}'.format(table))
    # app.info('Model built successfully')
    return model, code_vocab


model, code_vocab = load_model(trained_model='../pre_trained/models/all/', trained_vocab='../pre_trained/vocabs/')
"""

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
    from completion import run_completion
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

    if request.method == 'POST':
        data = request.get_data().decode("utf-8")
        print(args)
        predictions, prediction_scores = run_completion(
            args=args,
            source=data,
            trained_model=args.trained_model,
            trained_vocab=args.trained_vocab
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


'''
def handle_query():
    text = request.get_json()  # 获取用户query中的文本 例如"I love you"
    id_ = str(uuid.uuid1())  # 为query生成唯一标识
    d = {'id': id_, 'text': text}  # 绑定文本和query id
    redis_.rpush(db_key_query, json.dumps(d))  # 加入redis
    while True:
        result = redis_.get(id_)  # 获取该query的模型结果
        # print(result)
        if result is not None:
            redis_.delete(id_)
            result_text = {'code': "200", 'data': result}
            break
    return jsonify(result_text)  # 返回结果


def print_test(batch_size):
    while True:
        texts = []
        query_ids = []
        if redis_.llen(db_key_query) == 0:  # 若队列中没有元素就继续获取
            continue
        print(redis_.llen(db_key_query))
        for i in range(min(redis_.llen(db_key_query), batch_size)):
            query = redis_.lpop(db_key_query).decode('UTF-8')  # 获取query的text
            print(query)
            query_ids.append(json.loads(query)['id'])
            texts.append(json.loads(query)['text'])  # 拼接若干text 为batch
        result = {'label': 'POSITIVE', 'score': 0.99893874}  # 调用模型
        res = {'label': '', 'score': 0}
        for (id_, res) in zip(query_ids, result):
            res['score'] = str(res['score'])
            redis_.set(id_, json.dumps(res))  # 将模型结果送回队列
'''

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
