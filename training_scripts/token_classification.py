"""Fine-tune a BERT model on a token classification task
"""
import os
import sys
import json
import logging
from argparse import ArgumentParser, Namespace
from warnings import filterwarnings
from functools import partial
from itertools import chain
from collections import defaultdict
from datetime import datetime
from typing import Union, List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerFast,
    RobertaConfig,
    Trainer,
    TrainingArguments,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    set_seed
)
from datasets import load_dataset

here, _ = os.path.split(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(here, os.pardir, "src")))
from data_utils import (
    BatchEncodingDataset,
    load_kgi_model_checkpoint,
    load_kgi_tokenizer
)


TEXT_ENC = sys.getdefaultencoding()
LOGFMT = "%(asctime)s - %(levelname)s - \t%(message)s"

DATA_FP_HELP = """Data file path. Expects a folder containing train.json, dev.json and
test.json files, or a valid argument to pass to datasets.load_dataset, in the case
where the hfload option is activated."""
MODEL_HELP = """Path to a pre-trained PyTorch BERT-style model directory compatible with
the transformers token classification model."""
KGI_CP_HELP = """Indicates that `model` is a path to a UMLS-KGI checkpoint"""
MODEL_ATTR_HELP = """"""
LABEL_NAME_HELP = """The target class variable name in the dataset specified by data_fp
"""
TEXT_NAME_HELP = """The name of the text component in the input data file"""
SEQ_LEN_HELP = """Maximum number of tokens per sequence"""
TOKENIZER_HELP = """Optionally, specify a tokenizer to use"""
BATCH_SIZE_HELP = """Number of sequences to process at a time"""
EPOCHS_HELP = """Number of passes to run over the training & validation sets"""
STEPS_HELP = """Number of training updates to carry out; overwrites epochs"""
RUNS_HELP = """Number of total runs to carry out, varying the random seed each time"""
GRAD_ACC_HELP = """Number of training set batches over which to add up the gradient
between each backward pass"""
LR_HELP = """Learning rate to pass to the optimiser"""
WEIGHT_DECAY_HELP = """Decoupled weight decay value to apply in the Adam optimiser"""
OUTPUT_DIR_HELP = """Where to put the output directory"""
SEED_HELP = """Sets the base random state for the script, including the generation
of seeds for multiple runs"""
SUBSET_HELP = """Tells the HuggingFace datasets load function which version of the
dataset to load, where applicable"""
EVAL_SPLIT_NAME_HELP = """The name of the evaluation split in the input dataset
(this can vary - 'val', 'validation', 'dev', etc. - depending on the dataset)"""
CONST_SCHED_HELP = """Use a constant learning rate schedule; by default uses a
warmup of 500 or 1/5th of the training steps specified""" 
FILTER_LABELS_HELP = """If the dataset folder contains a file called `labels_to_use.txt`,
restrict the classification problem to only the class labels therein"""
NOEVAL_HELP = """Include this when the dataset only has a train/test split, without
a validation set (only applies to HuggingFace datasets)"""
SKIP_TEST_HELP = """Only run the training loop on the train & dev sets; for
debugging etc."""
HFLOAD_HELP = """Use the Huggingface datasets library to load the input dataset"""
BIO_HELP = """Add BIO schema formatting to the target labels"""
ADD_NONE_HELP = """Add a placeholder 'none' class to the target labels, which will
become class 0 in the collated input labels - for when not all of the tokens in the
input text have been labelled"""
AUTH_HELP = """Authorisation token for loading a model from the HuggingFace hub, if
required"""
NOSAVE_HELP = """Doesn't write anything to disk - can be useful for debugging"""
TKN_SENT_HELP = """Tells the tokenizer to consider individual strings in the input
dataset to be sentences - most NER datasets will already be formatted as lists
of individual words/tokens, but use this flag in cases where they aren't"""
METRIC_AVG = "micro", "macro", "weighted"
METRICS = "_precision", "_recall", "_f1"


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("data_fp", type=str, help=DATA_FP_HELP)
    parser.add_argument("model", type=str, help=MODEL_HELP)
    parser.add_argument("--from_kgi_checkpoint", action="store_true", help=KGI_CP_HELP)
    parser.add_argument("--model_attr", type=str, default="transformer", help=MODEL_ATTR_HELP)
    parser.add_argument("--label_name", type=str, default="tag", help=LABEL_NAME_HELP)
    parser.add_argument("--text_name", type=str, default="text", help=TEXT_NAME_HELP)
    parser.add_argument("--seq_len", type=int, default=512, help=SEQ_LEN_HELP)
    parser.add_argument("--tokenizer_path", type=str, help=TOKENIZER_HELP)
    parser.add_argument("--batch_size", type=int, default=16, help=BATCH_SIZE_HELP)
    parser.add_argument("--epochs", type=int, default=4, help=EPOCHS_HELP)
    parser.add_argument("--steps", type=int, default=0, help=STEPS_HELP)
    parser.add_argument("--runs", type=int, default=4, help=RUNS_HELP)
    parser.add_argument("--grad_acc", type=int, default=1, help=GRAD_ACC_HELP)
    parser.add_argument("--lr", type=float, default=2e-5, help=LR_HELP)
    parser.add_argument("--weight_decay", type=float, default=.01, help=WEIGHT_DECAY_HELP)
    default_output_dir = os.path.join(os.getenv("HOME"), "token-clf-eval")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help=OUTPUT_DIR_HELP)
    parser.add_argument("--seed", type=int, default=42, help=SEED_HELP)
    parser.add_argument("--subset_name", type=str, help=SUBSET_HELP)
    parser.add_argument("--eval_split_name", type=str, default="validation",
        help=EVAL_SPLIT_NAME_HELP)
    parser.add_argument("--constant_schedule", action="store_true", help=CONST_SCHED_HELP)
    parser.add_argument("--filter_labels", action="store_true", help=FILTER_LABELS_HELP)
    parser.add_argument("--no_eval", action="store_true", help=NOEVAL_HELP)
    parser.add_argument("--skip_test", action="store_true", help=SKIP_TEST_HELP)
    parser.add_argument("--hfload", action="store_true", help=HFLOAD_HELP)
    parser.add_argument("--bio", action="store_true", help=BIO_HELP)
    parser.add_argument("--add_none", action="store_true", help=ADD_NONE_HELP)
    parser.add_argument("--auth", type=str, help=AUTH_HELP)
    parser.add_argument("--no_save", action="store_true", help=NOSAVE_HELP)
    parser.add_argument("--tokenize_sentences", action="store_true", help=TKN_SENT_HELP)
    return parser.parse_args()


def labels2bio(labels: List[str]) -> List[str]:
    """Converts a list of strings into BIO labels - essentially just doubles each label
    except for `none`"""
    res = []
    for label_list in labels:
        prev_label = None
        bio_labels = []
        for label in label_list:
            if label == "none":
                bio_labels.append("O")
            elif label == prev_label:
                bio_labels.append("I-" + label)
            else:
                bio_labels.append("B-" + label)
            prev_label = label
        res.append(bio_labels)
    return res


def load_ner_data(
    fp: Union[str, os.PathLike],
    label_name: str,
    text_name: str
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Returns two equally-sized lists of lists, extracted from the specified
    .json file"""
    if not os.path.isfile(fp):
        text = labels = None
    else:
        with open(fp, encoding=TEXT_ENC) as f_io:
            dataset = json.load(f_io)
        text, labels = [], []
        for doc in dataset.values():
            text.append(doc[text_name].replace("\n", " ").split() if \
                isinstance(doc[text_name], str) else doc[text_name])
            doc_labels = doc[label_name]
            if isinstance(doc_labels, list):
                labels.append(doc_labels)
            elif isinstance(doc_labels, str):
                labels.append(doc_labels.split())
    return text, labels


def make_word_ids(
    input_ids: List[int], attn_mask: List[int],
    tokenizer: PreTrainedTokenizerFast, text_: List[str],
    start_from: int=0
) -> List[int]:
    """Workaround method for slow tokenizers"""
    word_ids = [None]  # sequence will always start with cls
    current_id = start_from
    for ii, attn in zip(input_ids[1:], attn_mask[1:]):
        if not attn:
            word_ids.append(None)
            continue
        try:
            next_word = text_[current_id + 1]
        except IndexError:
            word_ids.append(None)
            continue
        if next_word.startswith(tokenizer.decode(ii)):
            # start of the next word
            current_id += 1
        word_ids.append(current_id)
    return word_ids


def tokenize_and_align(
    text: List[List[str]], labels: List[int], tokenizer: PreTrainedTokenizerFast,
    label2id: Dict[str, int], is_split_into_words: bool
) -> BatchEncoding:
    """Runs tokenisation on the given text and aligns the target labels with the resulting
    subword identifiers"""
    if not text or not labels:
        return
    input_encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        is_split_into_words=is_split_into_words,
        return_overflowing_tokens=True
    )
    aligned_labels = []
    if not tokenizer.is_fast and any(input_encoding["overflowing_tokens"]):
        n_overflows_added = 0
        for i, overflow in enumerate(input_encoding["overflowing_tokens"]):
            if not overflow:
                continue
            overflow_idx = 0
            while overflow_idx < len(overflow):
                overflow_ = overflow[overflow_idx:]
                overflow_.insert(0, tokenizer.bos_token_id)
                attn_mask_ = np.ones(len(overflow_)).tolist()
                overflow_.insert(tokenizer.model_max_length, tokenizer.sep_token_id)
                if len(overflow_) < tokenizer.model_max_length:
                    while len(overflow_) < tokenizer.model_max_length:
                        overflow_.append(tokenizer.pad_token_id)
                    while len(attn_mask_) < tokenizer.model_max_length:
                        attn_mask_.append(0)
                insertion_idx = i + n_overflows_added + 1
                input_encoding["input_ids"].insert(insertion_idx, overflow_[:tokenizer.model_max_length])
                input_encoding["attention_mask"].insert(insertion_idx, attn_mask_)
                n_overflows_added += 1
                overflow_idx += tokenizer.model_max_length - 2
    n_sequences = len(input_encoding["input_ids"])
    original_idx = -1
    for i in range(n_sequences):
        if not tokenizer.is_fast:
            input_ids, attn_mask = input_encoding["input_ids"][i], \
                input_encoding["attention_mask"][i]
            text_ = text[i] if is_split_into_words else text[i].split()
            word_ids = make_word_ids(input_ids, attn_mask, tokenizer, text_)
        else:
            word_ids = input_encoding.word_ids(batch_index=i)
        if word_ids[1] <= 1:
           original_idx += 1
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                # special tokens
                label_ids.append(-100)
            elif word_id != prev_word_id:
                # first token of the word
                try:
                    k = labels[original_idx][word_id]
                    if not isinstance(k, str):
                        k = str(k)
                    id_ = label2id[k]
                except KeyError:
                    # label in the dev set that's not in the training set; ignore
                    id_ = -100
                label_ids.append(id_)
            else:
                # subsequent subword tokens
                label_ids.append(-100)
            prev_word_id = word_id
        aligned_labels.append(label_ids)
    input_encoding["labels"] = aligned_labels
    return input_encoding


def main(args: Namespace, logger: logging.Logger) -> None:
    if "/" in args.model:
        _, output_name_model = os.path.split(args.model)
    else:
        output_name_model = args.model
    if not args.no_save:
        _, output_name_data = os.path.split(args.data_fp if args.data_fp[-1] != "/" else args.data_fp[:-1])
        now = datetime.now()
        output_subdir = f"{output_name_model}_{output_name_data}_{now.day}-{now.month}_{now.hour}-{now.minute}"
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        output_dir = os.path.join(args.output_dir, output_subdir)
        os.mkdir(output_dir)
        with open(os.path.join(output_dir, "cl_args.json"), "w", encoding=TEXT_ENC) as f_io:
            json.dump(vars(args), f_io)
        logger.addHandler(logging.FileHandler(os.path.join(output_dir, "logger_output.log")))
    else:
        output_dir = None
    logger.info("Loading dataset from %s", args.data_fp)
    if args.hfload:
        hfload_train_kwargs = {"split": "train"}
        if args.subset_name:
            hfload_train_kwargs["name"] = args.subset_name
        train_data = load_dataset(args.data_fp, **hfload_train_kwargs)
        text_key = args.text_name if args.text_name is not None else "tokens"
        train_text, train_labels = train_data[text_key], train_data[args.label_name]
        if args.no_eval:
            dev_text = dev_labels = None
        else:
            hfload_dev_kwargs = {"split": args.eval_split_name}
            if args.subset_name:
                hfload_dev_kwargs["name"] = args.subset_name
            dev_data = load_dataset(args.data_fp, **hfload_dev_kwargs)
            dev_text, dev_labels = dev_data[text_key], dev_data[args.label_name]
    else:
        train_data, dev_data = map(
            lambda x: load_ner_data(
                os.path.join(args.data_fp, x + ".json"),
                label_name=args.label_name, text_name=args.text_name
            ), ("train", "dev")
        )
        train_text, train_labels = train_data
        dev_text, dev_labels = dev_data

    # extract labels & map to ints: starting with none then alphabetical order
    label_values = []
    for label_list in train_labels:
        labelset = set(label_list)
        for label in labelset:
            membership_check = label not in label_values
            if args.bio:
                membership_check = membership_check and label != "none"
            if membership_check:
                label_values.append(label)
    if args.filter_labels and os.path.isfile(os.path.join(args.data_fp, "labels_to_use.txt")):
        with open(os.path.join(args.data_fp, "labels_to_use.txt"), encoding=TEXT_ENC) as f_io:
            labels_to_use = set(f_io.read().split("\n"))
        label_values = list(set(label_values).intersection(labels_to_use))
    labels_are_strings = [isinstance(val, str) for val in label_values]
    if any(labels_are_strings):
        if not all(labels_are_strings):
            label_values = [str(val) for val in label_values]
        label_values.sort()
        if args.bio:
            id_ = 0
            label2id = {"O": id_}
            for label_value in label_values:
                id_ += 1
                label2id["B-" + label_value] = id_
                id_ += 1
                label2id["I-" + label_value] = id_
            train_labels, dev_labels = map(labels2bio, (train_labels, dev_labels))
        else:
            label_values.sort()
            label2id = {
                label_value: i for i, label_value in enumerate(
                    ["none", *label_values] if args.add_none else label_values
                )
            }
    else:
        label_values.sort()
        label2id = {str(i): i for i in label_values}
    id2label = {v: k for k, v in label2id.items()}
    model_init_kwargs = {"num_labels": len(label2id), "id2label": id2label, "label2id": label2id}
    logger.info("Model setup...")
    if args.auth is not None:
        auth_kwargs = {"use_auth_token": args.auth, "trust_remote_code": True}
        model_init_kwargs.update(auth_kwargs)
    if args.from_kgi_checkpoint:
        if not args.tokenizer_path:
            # assume the model path is a checkpoint subdirectory of the standard KGI pretraining output
            # and if we go back up a step we'll get the original tokenizer path from the input arguments
            with open(
                os.path.normpath(os.path.join(args.model, os.pardir, "script_params.json")),
                encoding=TEXT_ENC
            ) as f_io:
                params = json.load(f_io)
                tokenizer_path = params.get("tokenizer_path")
                if tokenizer_path is None:
                    tokenizer_path = params.get("model_path")
                args.tokenizer_path = tokenizer_path
        is_local_tokenizer = os.path.isfile(args.tokenizer_path)
        tokenizer = load_kgi_tokenizer(
            args.tokenizer_path,
            add_bert_special_tokens=is_local_tokenizer,
            load_via_hf=not is_local_tokenizer,
            # relation_tokens=REL_TOKENS if is_local_tokenizer else [],
            relation_tokens=[],
            model_max_length=args.seq_len
        )
        model = load_kgi_model_checkpoint(
            filepath=args.model,
            model_class=AutoModelForTokenClassification,
            transformer_model_attr=args.model_attr,
            remove_modules_from_transformer=["pooler"],
            vocab_size=tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") \
                else tokenizer.get_vocab().__len__(),
            model_init_kwargs=model_init_kwargs
        )
    else:
        tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model
        if os.path.isfile(tokenizer_path):
            tokenizer = load_kgi_tokenizer(
                tokenizer_path,
                add_bert_special_tokens=True,
                load_via_hf=False,
                relation_tokens=[],
                model_max_length=args.seq_len
            )
        else:
            kwargs = {} if args.auth is None else auth_kwargs
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
        model = AutoModelForTokenClassification.from_pretrained(args.model, **model_init_kwargs)

    optimizer = AdamW(tuple(model.parameters()), lr=args.lr)
    if args.constant_schedule:
        scheduler = get_constant_schedule(optimizer)
    else:
        warmup = int(.2 * args.steps) if args.steps else 500
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup)

    logger.info("Tokenizing text & constructing encoded datasets...")
    set_seed(args.seed)
    if hasattr(tokenizer, "max_length"):
        tokenizer.max_length = args.seq_len
    else:
        tokenizer.model_max_length = args.seq_len
    input_encoding_train, input_encoding_dev = map(
        partial(tokenize_and_align, tokenizer=tokenizer, label2id=label2id,
            is_split_into_words=not args.tokenize_sentences),
        (train_text, dev_text), (train_labels, dev_labels)
    )
    train_dataset = BatchEncodingDataset(input_encoding_train)
    if input_encoding_dev:
        dev_dataset = BatchEncodingDataset(input_encoding_dev)
        evaluation_strategy = "epoch"
    else:
        dev_dataset = None
        evaluation_strategy = "no"
    def compute_metrics(input_):
        clf_output_predictions, labels = input_
        class_predictions = np.argmax(clf_output_predictions, axis=2)
        per_sequence_metrics = defaultdict(list)
        for prediction, label in zip(class_predictions, labels):
            preds, refs = [], []
            for pred, lbl in zip(prediction, label):
                if lbl != -100:
                    preds.append(id2label[pred])
                    refs.append(id2label[lbl])
            for average in METRIC_AVG:
                metric_values = precision_recall_fscore_support(
                    refs, preds, average=average
                )
                for name, val in zip(METRICS, metric_values):
                    per_sequence_metrics[average + name].append(0 if np.isnan(val) else val)
        return {k: sum(v) / len(v) for k, v in per_sequence_metrics.items()}

    per_run_results = defaultdict(list)
    rng = np.random.default_rng(seed=args.seed)
    seeds = rng.integers(0, np.iinfo(np.int16).max, size=args.runs, dtype=np.int16)
    metric_names = {avg + met for met in METRICS for avg in METRIC_AVG}
    if args.no_save:
        save_strategy = "no"
    else:
        save_strategy = "steps" if args.steps else "epoch"
    for run, seed in enumerate(seeds):
        torch.manual_seed(seed)
        run_output_dir = os.path.join(output_dir if output_dir else "", "run" + str(run + 1))
        train_args = TrainingArguments(
            output_dir=run_output_dir,
            weight_decay=args.weight_decay,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_acc,
            learning_rate=args.lr,
            warmup_ratio=.1,
            num_train_epochs=args.epochs,
            max_steps=args.steps,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            save_steps=args.steps,
            fp16=True,
            seed=seed
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            optimizers=(optimizer, scheduler),
            compute_metrics=None if args.no_eval else compute_metrics
        )
        logger.info("Launching training run %d...", run + 1)
        trainer.train()

        if not args.skip_test:
            logger.info("Loading & encoding test dataset...")
            if args.hfload:
                test_data = load_dataset(args.data_fp, split="test")
                test_text, test_labels = test_data[text_key], test_data[args.label_name]
            else:
                test_text, test_labels = load_ner_data(
                    os.path.join(args.data_fp, "test.json"),
                    args.label_name, args.text_name
                )
            test_encoding = tokenize_and_align(
                test_text, test_labels, tokenizer, label2id,
                is_split_into_words=not args.tokenize_sentences
            )
            logger.info("Running model on test dataset...")
            test_dataset = BatchEncodingDataset(test_encoding)
            if args.no_eval:
                trainer.compute_metrics = compute_metrics
            test_results = trainer.evaluate(test_dataset)
            with open(os.path.join(run_output_dir, "results.json"), "w", encoding=TEXT_ENC) as f_io:
                json.dump(test_results, f_io)
            for k, v in test_results.items():
                metric_name = k.replace("eval_", "")
                if metric_name in metric_names:
                    per_run_results[metric_name].append(v)
    averaged_results = {}
    if per_run_results:
        for k, v in per_run_results.items():
            arr = np.array(v)
            averaged_results[k] = {"mean": arr.mean(), "std": arr.std()}
        logger.info(
            "-- Test Set Results: averaged over %d runs -- \
            \nMicro: P=%.5f +/- %.5f, R=%.5f +/- %.5f, F=%.5f +/- %.5f \
            \nMacro: P=%.5f +/- %.5f, R=%.5f +/- %.5f, F=%.5f +/- %.5f \
            \nWeighted: P=%.5f +/- %.5f, R=%.5f +/- %.5f, F=%.5f +/- %.5f", args.runs,
            *list(chain(*[v.values() for v in averaged_results.values()]))
        )
    if output_dir:
        with open(os.path.join(output_dir, "results.json"), "w", encoding=TEXT_ENC) as f_io:
            json.dump(averaged_results, f_io)
        with open(os.path.join(output_dir, "run_seeds.txt"), "w", encoding=TEXT_ENC) as f_io:
            f_io.write("\n".join(seeds.astype(str).tolist()))
        logger.info("Done! Files in %s", output_dir)
    else:
        logger.info("Done!")


if __name__ == "__main__":
    logger_ = logging.getLogger(__name__)
    logging.basicConfig(
        format=LOGFMT,
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO
    )
    filterwarnings(action="ignore", category=UserWarning)
    args_ = parse_arguments()
    main(args_, logger_)
