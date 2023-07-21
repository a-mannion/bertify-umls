"""Fine-tune a BERT model on a token classification task!
"""
import os
import sys
import json
import logging
from argparse import ArgumentParser
from warnings import filterwarnings
from functools import partial
from itertools import chain
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import wandb
from torch.utils.data import Dataset
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    Trainer,
    TrainingArguments,
    get_constant_schedule,
    set_seed
)
from transformers.integrations import WandbCallback
from datasets import load_dataset

here, _ = os.path.split(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(here, os.pardir, "src")))
from data_utils import BatchEncodingDataset


TEXT_ENC = sys.getdefaultencoding()
LOGFMT = "%(asctime)s - %(levelname)s - \t%(message)s"

DATA_FP_HELP = """Data file path. Expects a folder containing train.json, dev.json and
test.json files, or a valid argument to pass to datasets.load_dataset, in the case
where the hfload option is activated."""
MODEL_HELP = """Path to a pre-trained PyTorch BERT-style model directory compatible with
the transformers token classification model."""
KGI_CP_HELP = """Indicates that `model` is a path to a UMLS-KGI checkpoint"""
LABEL_NAME_HELP = """The target class variable name in the dataset specified by data_fp
"""
TEXT_NAME_HELP = """The name of the text component in the input data file"""
SEQ_LEN_HELP = """Maximum number of tokens per sequence"""
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
EVAL_SPLIT_NAME_HELP = """The name of the evaluation split in the input dataset
(this can vary - 'val', 'validation', 'dev', etc. - depending on the dataset)"""
FILTER_LABELS_HELP = """If the dataset folder contains a file called `labels_to_use.txt`,
restrict the classification problem to only the class labels therein"""
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
WBW_HELP = """Include this flag when using models implementing slow tokenizers
that cannot be automatically converted to fast ones - it tells the data loading
function to align tokens with labels itself instead of using built-in tokenizers
functionality"""
PROJ_HELP = """Name of the Weights & Biases project to log progress to"""
ENT_HELP = """Your Weights & Biases username; if this and/or `wandb_proj` are not
provided, the script will run locally without logging metrics"""
METRIC_AVG = "micro", "macro", "weighted"
METRICS = "_precision", "_recall", "_f1"


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("data_fp", type=str, help=DATA_FP_HELP)
    parser.add_argument("model", type=str, help=MODEL_HELP)
    parser.add_argument("--from_kgi_checkpoint", action="store_true", help=KGI_CP_HELP)
    parser.add_argument("--label_name", type=str, default="pos_tag", help=LABEL_NAME_HELP)
    parser.add_argument("--text_name", type=str, default="text", help=TEXT_NAME_HELP)
    parser.add_argument("--seq_len", type=int, default=512, help=SEQ_LEN_HELP)
    parser.add_argument("--batch_size", type=int, default=16, help=BATCH_SIZE_HELP)
    parser.add_argument("--epochs", type=int, default=4, help=EPOCHS_HELP)
    parser.add_argument("--steps", type=int, default=-1, help=STEPS_HELP)
    parser.add_argument("--runs", type=int, default=4, help=RUNS_HELP)
    parser.add_argument("--grad_acc", type=int, default=1, help=GRAD_ACC_HELP)
    parser.add_argument("--lr", type=float, default=2e-5, help=LR_HELP)
    parser.add_argument("--weight_decay", type=float, default=.01, help=WEIGHT_DECAY_HELP)
    default_output_dir = os.path.join(os.getenv("HOME"), "token-clf-eval")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help=OUTPUT_DIR_HELP)
    parser.add_argument("--seed", type=int, default=42, help=SEED_HELP)
    parser.add_argument("--eval_split_name", type=str, default="validation",
        help=EVAL_SPLIT_NAME_HELP)
    parser.add_argument("--filter_labels", action="store_true", help=FILTER_LABELS_HELP)
    parser.add_argument("--skip_test", action="store_true", help=SKIP_TEST_HELP)
    parser.add_argument("--hfload", action="store_true", help=HFLOAD_HELP)
    parser.add_argument("--bio", action="store_true", help=BIO_HELP)
    parser.add_argument("--add_none", action="store_true", help=ADD_NONE_HELP)
    parser.add_argument("--auth", type=str, help=AUTH_HELP)
    parser.add_argument("--no_save", action="store_true", help=NOSAVE_HELP)
    parser.add_argument("--tokenize_sentences", action="store_true", help=TKN_SENT_HELP)
    parser.add_argument("--wbw_alignment", action="store_true", help=WBW_HELP)
    parser.add_argument("--wandb_proj", type=str, help=PROJ_HELP)
    parser.add_argument("--wandb_entity", type=str, help=ENT_HELP)
    return parser.parse_args()


def labels2bio(labels):
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


def load_ner_data(fp, label_name, text_name, split_into_words):
    """Returns two equally-sized lists of lists, extracted from the specified
    .json file"""
    with open(fp, encoding=TEXT_ENC) as f_io:
        dataset = json.load(f_io)
    text, labels = [], []
    for doc in dataset.values():
        text.append(doc[text_name].split() if split_into_words else doc[text_name])
        doc_labels = doc[label_name]
        if isinstance(doc_labels, list):
            labels.append(doc_labels)
        elif isinstance(doc_labels, str):
            labels.append(doc_labels.split())
    return text, labels


def make_word_ids(input_ids, attn_mask, tokenizer, text_):
    """Workaround method for slow tokenizers"""
    word_ids = [None]  # sequence will always start with cls
    current_id = 0
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


def tokenize_and_align(text, labels, tokenizer, label2id, is_split_into_words, manual_word_ids=False):
    """Runs tokenisation on the given text and aligns the target labels with the resulting
    subword identifiers """
    input_encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        is_split_into_words=is_split_into_words
    )
    aligned_labels = []
    for i, label in enumerate(labels):
        if manual_word_ids:
            text_ = text[i] if is_split_into_words else text[i].split()
            input_ids, attn_mask = input_encoding["input_ids"][i], \
                input_encoding["attention_mask"][i]
            word_ids = make_word_ids(input_ids, attn_mask, tokenizer, text_)
        else:
            word_ids = input_encoding.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                # special tokens
                label_ids.append(-100)
            elif word_id != prev_word_id:
                # first token of the word
                try:
                    k = label[word_id]
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


def main(args, use_wandb):
    logger.info("Loading dataset from %s", args.data_fp)
    if args.hfload:
        train_data = load_dataset(args.data_fp, split="train")
        dev_data = load_dataset(args.data_fp, split=args.eval_split_name)
        # text_key = args.hfload_text_name if args.hfload_text_name is not None else "tokens"
        train_text, train_labels = train_data["tokens"], train_data[args.label_name]
        dev_text, dev_labels = dev_data["tokens"], dev_data[args.label_name]
    else:
        train_data, dev_data = map(
            lambda x: load_ner_data(
                os.path.join(args.data_fp, x + ".json"),
                label_name=args.label_name, text_name=args.text_name,
                split_into_words=not args.tokenize_sentences
            ), ("train", "dev")
        )
        train_text, train_labels = train_data
        dev_text, dev_labels = dev_data

    # extract labels & map to ints: starting with none then alphabetical order
    label_values = []
    for label_list in train_labels:
        labelset = set(label_list)
        for label in labelset:
            if label not in label_values and label != "none":
                label_values.append(label)
    if args.filter_labels and os.path.isfile(os.path.join(args.data_fp, "labels_to_use.txt")):
        with open(os.path.join(args.data_fp, "labels_to_use.txt"), encoding=TEXT_ENC) as f_io:
            labels_to_use = set(f_io.read().split("\n"))
        label_values = list(set(label_values).intersection(labels_to_use))
    label_values.sort()
    labels_are_strings = any(isinstance(val, str) for val in label_values)
    if labels_are_strings:
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
            label2id = {
                label_value: i for i, label_value in enumerate(
                    ["none", *label_values] if args.add_none else label_values
                )
            }
    else:
        label2id = {str(i): i for i in label_values}
    id2label = {v: k for k, v in label2id.items()}
    model_init_kwargs = {"num_labels": len(label2id), "id2label": id2label, "label2id": label2id}
    logger.info("Model setup...")
    if args.auth is not None:
        model_init_kwargs.update({"use_auth_token": args.auth, "trust_remote_code": True})
    if args.from_kgi_checkpoint:
        torch_dict = torch.load(os.path.join(args.model, "pytorch_model.bin"))
        model_init_kwargs["state_dict"] = {k.replace("transformer.", ""): v for k, v in torch_dict.items()}
    model = AutoModelForTokenClassification.from_pretrained(args.model, **model_init_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    optimizer = AdamW(tuple(model.parameters()), lr=args.lr)
    scheduler = get_constant_schedule(optimizer)

    logger.info("Tokenizing text & constructing encoded datasets...")
    set_seed(args.seed)
    if hasattr(tokenizer, "max_length"):
        tokenizer.max_length = args.seq_len
    else:
        tokenizer.model_max_length = args.seq_len
    input_encoding_train, input_encoding_dev = map(
        partial(tokenize_and_align, tokenizer=tokenizer, label2id=label2id,
            is_split_into_words=not args.tokenize_sentences, manual_word_ids=args.wbw_alignment),
        (train_text, dev_text), (train_labels, dev_labels)
    )
    train_dataset = BatchEncodingDataset(input_encoding_train)
    dev_dataset = BatchEncodingDataset(input_encoding_dev)

    logger.info("Training setup...")
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

    if "/" in args.model:
        _, output_name_model = os.path.split(args.model)
    else:
        output_name_model = args.model
    _, output_name_data = os.path.split(args.data_fp if args.data_fp[-1] != "/" else args.data_fp[:-1])
    now = datetime.now()
    output_subdir = f"{output_name_model}_{output_name_data}_{now.day}-{now.month}_{now.hour}-{now.minute}"
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    output_dir = os.path.join(args.output_dir, output_subdir)
    os.mkdir(output_dir)
    with open(os.path.join(output_dir, "cl_args.json"), "w", encoding=TEXT_ENC) as f_io:
        json.dump(vars(args), f_io)
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
        run_output_dir = os.path.join(output_dir, "run" + str(run + 1))
        train_args = TrainingArguments(
            output_dir=run_output_dir,
            weight_decay=args.weight_decay,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_acc,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            max_steps=args.steps,
            evaluation_strategy="epoch",
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
            compute_metrics=compute_metrics
        )
        logger.info("Launching training run %d...", run + 1)
        if not use_wandb:
            trainer.remove_callback(WandbCallback)
        trainer.train()

        if not args.skip_test:
            logger.info("Loading & encoding test dataset...")
            if args.hfload:
                test_data = load_dataset(args.data_fp, split="test")
                test_text, test_labels = test_data["tokens"], test_data[args.label_name]
            else:
                test_text, test_labels = load_ner_data(
                    os.path.join(args.data_fp, "test.json"),
                    args.label_name, args.text_name,
                    split_into_words=not args.tokenize_sentences
                )
            test_encoding = tokenize_and_align(
                test_text, test_labels, tokenizer, label2id,
                is_split_into_words=not args.tokenize_sentences,
                manual_word_ids=args.wbw_alignment
            )
            logger.info("Running model on test dataset...")
            test_dataset = BatchEncodingDataset(test_encoding)
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
    with open(os.path.join(output_dir, "results.json"), "w", encoding=TEXT_ENC) as f_io:
        json.dump(averaged_results, f_io)
    with open(os.path.join(output_dir, "run_seeds.txt"), "w", encoding=TEXT_ENC) as f_io:
        f_io.write("\n".join(seeds.astype(str).tolist()))
    logger.info("Done! Files in %s", output_dir)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format=LOGFMT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)
    filterwarnings(action="ignore", category=UserWarning)
    args_ = parse_arguments()
    use_wandb_ = args_.wandb_proj and args_.wandb_entity
    if use_wandb_:
        with wandb.init(project=args_.wandb_proj, entity=args_.wandb_entity, config=vars(args_)):
            main(args_, use_wandb_)
    else:
        main(args_, use_wandb_)
