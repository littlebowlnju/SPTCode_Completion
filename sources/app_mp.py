import logging
from flask import Flask, request
import json
from concurrency.task import CodeCompletionTask
import time
import queue
import argparse
from args import add_args
from transformers import BartConfig
from data.vocab import load_vocab
from model.bart import BartForClassificationAndGeneration
import os
import enums
from completion import complete_batch
import threading
import torch


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

req_queue = queue.Queue()


@app.before_first_request
def poll():
    def check_queue():
        while True:
            if not req_queue.empty():
                print("req_queue size: ", req_queue.qsize())
                # batch size 4 is acceptable
                size = min(req_queue.qsize(), 4)
                task_list = []
                source_list = []
                for _ in range(size):
                    task = req_queue.get()
                    task_list.append(task)
                    source_list.append(task.code)

                # send the whole list to predict
                predictions, predictions_scores = complete_batch(
                    args=args,
                    source_list=source_list,
                    model=model,
                    code_vocab=code_vocab
                )
                assert len(predictions) == len(predictions_scores) == size
                for i in range(size):
                    pre_scores = []
                    for score in predictions_scores[i]:
                        pre_scores.append("%.2f%%" % (score * 100))
                    result = {
                        "predictions": predictions[i],
                        "prediction_scores": pre_scores
                    }
                    task_list[i].set_result(result)

            # polling every 0.01 seconds
            time.sleep(0.01)

    thread = threading.Thread(target=check_queue)
    thread.start()


@app.route("/", methods=["POST"])
def generate_result_mp():
    # print("------------------request received-------------------")
    data = request.get_data().decode("utf-8")
    task = CodeCompletionTask(data)
    req_queue.put(task)
    result = task.get_result()
    return json.dumps(result)


if __name__ == '__main__':
    # app.run(host="172.29.7.224", port=1818, debug=True)
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run()


