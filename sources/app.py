import argparse
import logging
import enums
import torch
from flask import Flask, jsonify, request
from args import add_args
from transformers import BartConfig, Seq2SeqTrainingArguments, EarlyStoppingCallback, \
    IntervalStrategy, SchedulerType
from data.vocab import Vocab, load_vocab, init_vocab
from model.bart import BartForClassificationAndGeneration
import os
import time
import json

logger = logging.getLogger(__name__)
app = Flask(__name__)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


@app.route("/", methods=['POST', 'GET'])
def generate_result():
    from completion import complete
    if request.method == 'POST':
        data = request.get_data().decode("utf-8")
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


if __name__ == '__main__':
    # app.run(host="172.29.7.224", port=1818, debug=True)
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)

    app.run()
