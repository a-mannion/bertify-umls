"""UMLS-KGI pre-training implementation
"""
import os
import sys
import json
import logging
from functools import partial
from datetime import datetime
from argparse import ArgumentParser, Namespace
from typing import Union
from tqdm import tqdm

import torch
import numpy as np
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    DistilBertConfig,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    set_seed
)
from torch.utils.data import DataLoader
from torch.optim import AdamW

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
)
from kgi_bert import KgiLMBert
from data_utils import (
    get_relation_labels,
    prepare_mixed_dataset,
    mixed_collate_fn
)

TEXT_ENC = sys.getdefaultencoding()
LOGFMT = "%(asctime)s - %(levelname)s - \t%(message)s"

KG_FP_HELP = """Path to the KGI dataset, as created by the `build_dataset` script"""
CORPUS_FP_HELP = """Path to a text file containing the training corpus for the
masked-language task"""
MODEL_PATH_HELP = """BERT encoder to use - can be a pre-trained checkpoint"""
TOKENIZER_PATH_HELP = """Tokenizer to use, if different from the one specified by
`model_path`"""
FROM_CP_HELP = """Include this flag when continuing training from a local UMLS-KGI checkpoint"""
FROM_SCRATCH_HELP = """Include this flag to use only the model config to instantiate the
transformer, i.e. randomly initialise the weights to train from scratch"""
ABST_HELP = """Add padding, masking, and separation tokens to the tokenizer during
data preprocessing"""
OUTPUT_DIR_HELP = """Where to put the output files"""
MODEL_RUN_NAME_HELP = """Name to use for the output directory"""
TRAINSETFRAC_HELP = "Proportion of the input dataset to use for training"
BATCH_SIZE_HELP = "Number of samples to process at a time"
LR_HELP = "Learning rate to use for optimisation"
EPOCHS_HELP = "Number of passes to run over the training set"
GRAD_ACC_HELP = "Number of batches over which to add up gradients between each backward pass"
LIN_SCHED_EPOCH_HELP = """For cases in which a non-linear learning rate schedule is being spread out
over multiple runs of this script, use this flag to indicate how many epochs have already been run
"""
SEQ_LEN_HELP = "Maximal number of tokens per training sequence"
SEED_HELP = "Random seed to use for initialisation"
TASK_COEFS_HELP = """Floating-point weights for the loss function components corresponding to each
KG-based task; expects three numbers, the first for entity prediction, the second for link
prediction, and the third for triple classification"""
CP_INTERVAL_HELP = "Save a checkpoint every `n` epochs"
N_TEXT_DOCS_HELP = """Specify a maximum number of documents to load from the `corpus_fp` file;
leave unspecified to use all of them"""
FP16_HELP = """Use 16-bit precision training"""
CONST_SCHED_HELP = """Use a constant learning rate"""
TDO_HELP = """Only update model weights based on the training data - otherwise, the model will be
trained on the validation set AFTER the standard training run; use this flag when you intend to
continue training the model on the same dataset later"""
NOSAVE_HELP = """Doesn't write anything to disk - can be useful for debugging"""
TQDM_HELP = "Show a progress bar tracking each epoch over the number of batches"


def parse_arguments() -> Namespace:
    """Command line arguments"""
    parser = ArgumentParser()
    parser.add_argument("kg_fp", type=str, help=KG_FP_HELP)
    parser.add_argument("corpus_fp", type=str, help=CORPUS_FP_HELP)
    parser.add_argument("model_path", type=str, help=MODEL_PATH_HELP)
    parser.add_argument("--tokenizer_path", type=str, help=TOKENIZER_PATH_HELP)
    parser.add_argument("--from_cp", action="store_true", help=FROM_CP_HELP)
    parser.add_argument("--from_scratch", action="store_true", help=FROM_SCRATCH_HELP)
    parser.add_argument("--add_bert_special_tokens", action="store_true", help=ABST_HELP)
    parser.add_argument("--output_dir", type=str, help=OUTPUT_DIR_HELP)
    parser.add_argument("--model_run_name", type=str, default="kgi", help=MODEL_RUN_NAME_HELP)
    parser.add_argument("--train_set_frac", type=float, default=.95, help=TRAINSETFRAC_HELP)
    parser.add_argument("--batch_size", type=int, default=16, help=BATCH_SIZE_HELP)
    parser.add_argument("--lr", type=float, default=.00075, help=LR_HELP)
    parser.add_argument("--epochs", type=int, default=4, help=EPOCHS_HELP)
    parser.add_argument("--grad_acc", type=int, default=4, help=GRAD_ACC_HELP)
    parser.add_argument("--linear_schedule_epochs", type=int, help=LIN_SCHED_EPOCH_HELP)
    parser.add_argument("--seq_len", type=int, default=128, help=SEQ_LEN_HELP)
    parser.add_argument("--seed", type=int, default=42, help=SEED_HELP)
    parser.add_argument("--task_coefs", type=float, nargs=3, help=TASK_COEFS_HELP)
    parser.add_argument("--epoch_checkpoint_interval", type=int, default=4, help=CP_INTERVAL_HELP)
    parser.add_argument("--n_text_docs", type=int, help=N_TEXT_DOCS_HELP)
    parser.add_argument("--fp16", action="store_true", help=FP16_HELP)
    parser.add_argument("--constant_schedule", action="store_true", help=CONST_SCHED_HELP)
    parser.add_argument("--train_data_only", action="store_true", help=TDO_HELP)
    parser.add_argument("--nosave", action="store_true", help=NOSAVE_HELP)
    parser.add_argument("--pb", action="store_true", help=TQDM_HELP)
    return parser.parse_args()


def main(config: Namespace, logger: logging.Logger) -> None:
    # setup
    accelerator = Accelerator(
        mixed_precision="fp16" if config.fp16 else None,
        gradient_accumulation_steps=config.grad_acc
    )
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    set_seed(config.seed)

    # data loading/preprocessing/setup
    logger.info("Loading & processing KG dataset from %s and text corpus from %s",
        config.kg_fp, config.corpus_fp)
    if config.tokenizer_path is None:
        tokenizer_path = config.model_path
        load_tokenizer_via_hf = True
    else:
        tokenizer_path = config.tokenizer_path
        load_tokenizer_via_hf = not os.path.isfile(config.tokenizer_path)
    train_dataset, eval_dataset, tokenizer = prepare_mixed_dataset(
        kb_datadir=config.kg_fp,
        corpus_fp=config.corpus_fp,
        tokenizer=tokenizer_path,
        load_tokenizer_via_hf=load_tokenizer_via_hf,
        model_max_length=config.seq_len,
        n_text_docs=config.n_text_docs,
        train_set_frac=config.train_set_frac,
        add_bert_special_tokens=config.add_bert_special_tokens,
        shuffle=True
    )
    vocab_size = len(tokenizer)
    rel_token_ids2labels = get_relation_labels(tokenizer)
    collate_fn = partial(mixed_collate_fn, tokenizer=tokenizer, rel_token_ids2labels=rel_token_ids2labels)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, collate_fn=collate_fn)

    # model setup
    logger.info("Dataloaders prepared; setting up model: %s", config.model_path)
    if config.from_cp:
        model = KgiLMBert.from_pretrained(config.model_path)
    else:
        model_config = AutoConfig.from_pretrained(config.model_path, vocab_size=vocab_size)
        num_labels_link_pred = len(rel_token_ids2labels)
        if not config.task_coefs:
            _, task_idx_counts = np.unique(train_dataset.task_type_index, return_counts=True)
            kg_task_idx_counts = task_idx_counts[:-1]
            kg_task_idx_counts_sum = kg_task_idx_counts.sum()
            task_weight_coefficients = (kg_task_idx_counts_sum - kg_task_idx_counts) /\
                (2 * kg_task_idx_counts_sum)
            task_weight_coefficients[kg_task_idx_counts == 0] = 0
            config.task_coefs = task_weight_coefficients.tolist()
        model = KgiLMBert(
            model_config,
            from_pretrained=config.model_path if not config.from_scratch else None,
            num_labels_link_pred=num_labels_link_pred,
            task_weight_coefficients=config.task_coefs
        )
    optimizer = AdamW(tuple(model.parameters()), lr=config.lr)
    if config.from_cp:
        optimizer_state_dict = torch.load(os.path.join(config.model_path, "optimizer.bin"))
        optimizer.load_state_dict(optimizer_state_dict)

    if not config.nosave:
        if not args.output_dir:
            args.output_dir = os.path.join(os.getenv("HOME"), "umls-kgi-runs")
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        now = datetime.now()
        output_subdir = f"{config.model_run_name}_{now.day}-{now.month}_{now.hour}-{now.minute}"
        output_fp = os.path.join(args.output_dir, output_subdir)
        os.mkdir(output_fp)
        try:
            script_params = config.as_dict()
        except AttributeError:
            script_params = vars(config)
        with open(os.path.join(output_fp, "script_params.json"), "w", encoding=TEXT_ENC) as f_io:
            json.dump(script_params, f_io)

    n_train = len(train_dataset)
    n_train_batches = len(train_dataloader)
    optimizer_updates_per_epoch = int(n_train_batches / config.grad_acc)
    total_train_steps = config.epochs * optimizer_updates_per_epoch
    logger.info(
        "%d training sequences in %d batches, %d eval sequences in %d batches (d=%d)",
        n_train, n_train_batches, len(eval_dataset), len(eval_dataloader), config.seq_len
    )
    logger.info("N. epochs: %d", config.epochs)
    logger.info("Total optimisation steps: %d (accumulating gradients on %d batches at a time)",
        total_train_steps, config.grad_acc)
    logger.info(
        "Effective Batch Size: %d (%d x %d x %d)", 
        config.batch_size * config.grad_acc * accelerator.num_processes,
        config.batch_size, config.grad_acc, accelerator.num_processes
    )
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    logger.info("Model, optimiser & dataloaders wrapped in preparation for training")
    if config.constant_schedule:
        scheduler = get_constant_schedule(optimizer)
    elif config.linear_schedule_epochs is None:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(.2 * total_train_steps))
    else:
        # the parameter `config.linear_schedule_epochs` tells the scheduler how many epochs we intend to train this model for, irrespective of how many we're doing in this run
        # for continuing training from a checkpoint, going to need to give it the `last_epoch` parameter to tell it whereabouts in the schedule to resume
        num_training_steps = optimizer_updates_per_epoch * config.linear_schedule_epochs
        num_warmup_steps = int(.2 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    scheduler = accelerator.prepare(scheduler)
    if config.from_cp:
        accelerator.load_state(config.model_path)
    logger.info("Accelerator state - %s", repr(accelerator.state)[:-1].replace("\n", "; "))
    logger.info("=== Starting Training Loop ===")

    def _save_checkpoint():
        if accelerator.is_local_main_process:
            accelerator.wait_for_everyone()
            cp_dir = os.path.join(output_fp, f"checkpoint_epoch{epoch + 1}")
            if not config.nosave:
                try:
                    accelerator.save_state(cp_dir)
                    with open(os.path.join(cp_dir, "config.json"), "w", encoding=TEXT_ENC) as f_io:
                        json.dump(model.config.to_dict(), f_io)
                    with open(
                        os.path.join(cp_dir, "kgi_specific_config.json"), "w", encoding=TEXT_ENC
                    ) as f_io:
                        json.dump(model.kgi_specific_config, f_io)
                except Exception as exception:
                    logger.error(
                        "Checkpoint could not be saved because of the following error:\n%s",
                        exception
                    )
    
    def _run_epoch_iteration(mode, artefacts, dataloader, hide_progress_bar, accelerator=None):
        model, optimizer, scheduler = artefacts
        if mode == "train":
            model.train()
        else:
            model.eval()
        batch_loss = []
        for batch in tqdm(dataloader, disable=hide_progress_bar):
            if mode == "train":
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    accelerator.backward(outputs.loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                with torch.no_grad():
                    outputs = model(**batch)
            loss_val = outputs.loss.detach().item()
            if np.isnan(loss_val):  # usually underflow afaik
                loss_val = 1e-6
            batch_loss.append(loss_val)
        return sum(batch_loss) / len(batch_loss)

    train_loss_list, eval_loss_list = [], []
    hide_progress_bar = not config.pb
    for epoch in range(config.epochs):
        artefacts = model, optimizer, scheduler
        epoch_train_loss = _run_epoch_iteration(
            "train",
            artefacts,
            train_dataloader,
            hide_progress_bar,
            accelerator
        )
        train_loss_list.append(epoch_train_loss)

        epoch_eval_loss = _run_epoch_iteration(
            "eval", artefacts, eval_dataloader, hide_progress_bar
        )
        eval_loss_list.append(epoch_eval_loss)

        logger.info(
            "\tFinished epoch %d - train loss %.5f, eval loss %.5f",
            epoch + 1, epoch_train_loss, epoch_eval_loss
        )
        if (epoch + 1) % config.epoch_checkpoint_interval == 0 and epoch != config.epochs - 1:
            _save_checkpoint()

    if not config.train_data_only:
        logger.info("Training model on evaluation data...")
        torch.cuda.empty_cache()
        for epoch in range(config.epochs):
            model.train()
            for batch in eval_dataloader:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

    accelerator.wait_for_everyone()
    if not config.nosave:
        logger.info("All processes finished; saving model...")
        _save_checkpoint()
        with open(
            os.path.join(output_fp, "eval_loss_by_epoch.json"), "w", encoding=TEXT_ENC
        ) as f_io:
            json.dump({"loss": eval_loss_list}, f_io)
    logger.info("Done!")


if __name__ == "__main__":
    logger_ = logging.getLogger(__name__)
    logging.basicConfig(format=LOGFMT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)
    main(parse_arguments(), logger_)
    