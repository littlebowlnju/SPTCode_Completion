from typing import Union, Tuple
from transformers import BartConfig, Seq2SeqTrainingArguments, EarlyStoppingCallback, \
    IntervalStrategy, SchedulerType

import logging
import os
import torch

from data.vocab import Vocab, load_vocab, init_vocab
from model.bart import BartForClassificationAndGeneration
from utils.general import count_params, human_format, layer_wise_parameters
from eval.metrics import bleu, meteor, rouge_l, avg_ir_metrics, accuracy_for_sequence, accuracy_top_k_for_sequence
from utils.trainer import CodeTrainer
from utils.callbacks import LogStateCallBack
from data.data_utils import generate_input, my_parse_for_completion

MODEL_MODE_GEN = 'bart_gen'

logger = logging.getLogger(__name__)


def run_completion(
        args,
        source,
        trained_model: Union[BartForClassificationAndGeneration, str],
        trained_vocab: Union[Tuple[Vocab, Vocab, Vocab], str]
):
    """
    Run code completion, given source code, should give predictions list
    :param args: arguments
    :param source: str, source code string
    :param trained_model: trained model,
    :param trained_vocab: trained vocab
    :return: list[str], list of predictions with highest probability, len == 5
    """
    # -------------------------------
    # vocabs
    # -------------------------------
    if trained_vocab:
        if isinstance(trained_vocab, tuple):
            logger.info('Vocabularies are passed through parameter')
            assert len(trained_vocab) == 3
            code_vocab, ast_vocab, nl_vocab = trained_vocab
        else:
            logger.info('Loading vocabularies from files')
            code_vocab = load_vocab(vocab_root=trained_vocab, name='code')
            ast_vocab = load_vocab(vocab_root=trained_vocab, name='ast')
            nl_vocab = load_vocab(vocab_root=trained_vocab, name='nl')

    logger.info(f'The size of code vocabulary: {len(code_vocab)}')
    logger.info(f'The size of nl vocabulary: {len(nl_vocab)}')
    logger.info(f'The size of ast vocabulary: {len(ast_vocab)}')
    logger.info('Vocabularies built successfully')

    # ------------------------------
    # model
    # ------------------------------
    logger.info('-' * 100)
    if trained_model:
        if isinstance(trained_model, BartForClassificationAndGeneration):
            logger.info('Model is passed through parameter')
            model = trained_model
        else:
            logger.info('Loading the model from file')
            config = BartConfig.from_json_file(os.path.join(trained_model, 'config.json'))
            model = BartForClassificationAndGeneration.from_pretrained(os.path.join(trained_model, 'pytorch_model.bin'),
                                                                       config=config)
    model.set_model_mode(MODEL_MODE_GEN)
    # log model statistic
    logger.info('Trainable parameters: {}'.format(human_format(count_params(model))))
    table = layer_wise_parameters(model)
    logger.debug('Layer-wised trainable parameters:\n{}'.format(table))
    logger.info('Model built successfully')

    # --------------------------------------------------
    # trainer
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Initializing the running configurations')

    def decode_preds(preds):
        preds, labels = preds
        decoded_preds = code_vocab.decode_batch(preds)
        decoded_labels = code_vocab.decode_batch(labels)
        return decoded_labels, decoded_preds

    # compute metrics
    def compute_valid_metrics(eval_preds):
        decoded_labels, decoded_preds = decode_preds(eval_preds)
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result = {}
        result.update(bleu(references=refs, candidates=cans))
        result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    def compute_test_metrics(eval_preds):
        decoded_labels, decoded_preds = decode_preds(eval_preds)
        result = {'references': decoded_labels, 'candidates': decoded_preds}
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result.update(bleu(references=refs, candidates=cans))
        try:
            result.update(meteor(references=refs, candidates=cans))
        except Exception:
            pass
        result.update(rouge_l(references=refs, candidates=cans))
        result.update(avg_ir_metrics(references=refs, candidates=cans))
        result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.checkpoint_root, 'completion'),
                                             overwrite_output_dir=False,
                                             do_train=True,
                                             do_eval=True,
                                             do_predict=True,
                                             evaluation_strategy=IntervalStrategy.EPOCH,
                                             prediction_loss_only=False,
                                             per_device_train_batch_size=args.batch_size,
                                             per_device_eval_batch_size=args.eval_batch_size,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             learning_rate=args.learning_rate,
                                             weight_decay=args.lr_decay_rate,
                                             max_grad_norm=args.grad_clipping_norm,
                                             num_train_epochs=args.n_epoch,
                                             lr_scheduler_type=SchedulerType.LINEAR,
                                             warmup_steps=args.warmup_steps,
                                             logging_dir=os.path.join(args.tensor_board_root, 'completion'),
                                             logging_strategy=IntervalStrategy.STEPS,
                                             logging_steps=args.logging_steps,
                                             save_strategy=IntervalStrategy.EPOCH,
                                             save_total_limit=2,
                                             seed=args.random_seed,
                                             fp16=args.fp16,
                                             dataloader_drop_last=False,
                                             run_name=args.model_name,
                                             load_best_model_at_end=True,
                                             metric_for_best_model='accuracy',
                                             greater_is_better=True,
                                             ignore_data_skip=False,
                                             label_smoothing_factor=args.label_smoothing,
                                             report_to=['tensorboard'],
                                             dataloader_pin_memory=True,
                                             predict_with_generate=True)
    trainer = CodeTrainer(main_args=args,
                          code_vocab=code_vocab,
                          ast_vocab=ast_vocab,
                          nl_vocab=nl_vocab,
                          task='completion',
                          model=model,
                          args=training_args,
                          data_collator=None,
                          train_dataset=None,
                          eval_dataset=None,
                          tokenizer=nl_vocab,
                          model_init=None,
                          compute_metrics=compute_valid_metrics,
                          callbacks=[
                              EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience),
                              LogStateCallBack()])
    logger.info('Running configurations initialized successfully')

    # -----------------------------
    # generate model input from source code
    # -----------------------------
    inputs = generate_input(args, source)
    # inputs = my_parse_for_completion(args, './dataset/source.txt', './dataset/target.txt')

    # -----------------------------
    # predict
    # -----------------------------
    predictions = []
    prediction_scores = []
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
