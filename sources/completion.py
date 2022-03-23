import logging
import torch
from data.data_utils import generate_input, my_parse_for_completion

MODEL_MODE_GEN = 'bart_gen'

logger = logging.getLogger(__name__)


def complete(args, source, model, code_vocab):
    inputs = generate_input(args, source)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outputs = model.generate(
        input_ids=inputs['input_ids'].to(device),
        attention_mask=inputs['attention_mask'].to(device),
        max_length=args.completion_max_len,
        min_length=3,
        early_stopping=True,
        num_beams=args.beam_width,
        num_return_sequences=5,
        output_scores=True,
        return_dict_in_generate=True,
    )

    output_seqs = outputs.sequences
    output_seq_scores = outputs.sequences_scores
    output_seqs = output_seqs.view(1, -1, output_seqs.size(-1))
    for output in output_seqs:
        predictions = code_vocab.decode_batch(output.cpu().numpy())  # decode to strings
    print('--------------- predictions -----------------')
    print(predictions)
    print('--------------- predictions‘ scores ----------------')
    # score to softmax, transfer to probabilities sum to 1
    m = torch.nn.Softmax(dim=0)
    probabilities = m(output_seq_scores).cpu().numpy()

    # return scores as float
    prediction_scores = probabilities
    # Or transfer to percentage strings
    # for pro in probabilities:
    #     prediction_scores.append("%.2f%%" % (pro * 100))
    print(prediction_scores)
    return predictions, prediction_scores
