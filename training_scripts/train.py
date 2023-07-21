"""UMLS-KGI pre-training implementation
"""
import os
import sys
import json
import logging
from functools import partial
from datetime import datetime
from argparse import ArgumentParser

import torch
import wandb
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

here, _ = os.path.split(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(here, os.pardir, "src")))
from kgi_bert import KgiLMBert
from data_utils import (
    Bunch,
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
FROM_SCRATCH_HELP = """Include this flag to use only the model config to instantiate the transformer,
i.e. randomly initialise the weights to train from scratch"""
ABST_HELP = """Manually add padding, masking, and separation tokens to the tokenizer during 
data preprocessing"""
MODEL_RUN_NAME_HELP = """Name to use for the output directory"""
CONFIG_FILE_HELP = """Path to a json file containing additional arguments, see `config_files/kgi_config.json`"""
N_TEXT_DOCS_HELP = """Specify a maximum number of documents to load from the `corpus_fp` file; leave unspecified
to use all of them"""
EXCLUDE_TASK_HELP = """Integer argument telling the model to ignore one of the training objectives (for ablation
experiments) - 0 for entity prediction, 1 for link prediction and 2 for triple classification"""
FP16_HELP = """Use 16-bit precision training"""
CONST_SCHED_HELP = """Use a constant learning rate"""
AUTOCOEF_HELP = """Automatically weight the loss function components for each sub-task based on the number of 
available training examples for each one - takes precedence over the values specified in `config_file`"""
TDO_HELP = """Only update model weights based on the training data - otherwise, the model will  be trained on the
validation set AFTER the standard training run; use this flag when you intend to continue training the model on
the same dataset later"""
NOSAVE_HELP = """Doesn't write anything to disk - can be useful for debugging"""
PROJ_HELP = """Name of the Weights & Biases project to log progress to"""
ENT_HELP = """Your Weights & Biases username; if this and/or `wandb_proj` are not provided, the script will run
locally without logging metrics"""


def parse_arguments():
    """Command line arguments"""
    parser = ArgumentParser()
    parser.add_argument("kg_fp", type=str, help=KG_FP_HELP)
    parser.add_argument("corpus_fp", type=str, help=CORPUS_FP_HELP)
    parser.add_argument("model_path", type=str, help=MODEL_PATH_HELP)
    parser.add_argument("--tokenizer_path", type=str, help=TOKENIZER_PATH_HELP)
    parser.add_argument("--from_cp", action="store_true", help=FROM_CP_HELP)
    parser.add_argument("--from_scratch", action="store_true", help=FROM_SCRATCH_HELP)
    parser.add_argument("--add_bert_special_tokens", action="store_true", help=ABST_HELP)
    parser.add_argument("--model_run_name", type=str, default="kgi", help=MODEL_RUN_NAME_HELP)
    default_config_file = os.path.join(here, "config_files/kgi_config.json")
    parser.add_argument("--config_file", type=str, default=default_config_file, help=CONFIG_FILE_HELP)
    parser.add_argument("--n_text_docs", type=int, help=N_TEXT_DOCS_HELP)
    parser.add_argument("--exclude_task", type=int, choices={0, 1, 2}, help=EXCLUDE_TASK_HELP)
    parser.add_argument("--fp16", action="store_true", help=FP16_HELP)
    parser.add_argument("--constant_schedule", action="store_true", help=CONST_SCHED_HELP)
    parser.add_argument("--auto_coef", action="store_true", help=AUTOCOEF_HELP)
    parser.add_argument("--train_data_only", action="store_true", help=TDO_HELP)
    parser.add_argument("--nosave", action="store_true", help=NOSAVE_HELP)
    parser.add_argument("--wandb_proj", type=str, help=PROJ_HELP)
    parser.add_argument("--wandb_entity", type=str, help=ENT_HELP)
    return parser.parse_args()


def setup_metric_step_axes():
    # define x-axis for graphs
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    # associate output metrics with the corresponding x-axis metric
    wandb.define_metric("train_loss", step_metric="train_step")
    wandb.define_metric("eval_loss", step_metric="eval_step")


def run_pipeline(config):
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
    rel_token_ids2labels = get_relation_labels(tokenizer)
    collate_fn = partial(mixed_collate_fn, tokenizer=tokenizer, rel_token_ids2labels=rel_token_ids2labels)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, collate_fn=collate_fn)

    # model setup
    logger.info("Dataloaders prepared; setting up model: %s", config.model_path)
    if config.from_cp:
        model = KgiLMBert.from_pretrained(config.model_path)
    else:
        try:
            vocab_size = tokenizer.get_vocab_size()
        except AttributeError:
            vocab_size = len(tokenizer.get_vocab())
        model_config = AutoConfig.from_pretrained(config.model_path, vocab_size=vocab_size)
        
        num_labels_link_pred = len(rel_token_ids2labels)
        if config.auto_coef:
            _, task_idx_counts = np.unique(train_dataset.task_type_index, return_counts=True)
            kg_task_idx_counts = task_idx_counts[:-1]
            kg_task_idx_counts_sum = kg_task_idx_counts.sum()
            task_weight_coefficients = (kg_task_idx_counts_sum - kg_task_idx_counts) /\
                (2 * kg_task_idx_counts_sum)
            task_weight_coefficients[kg_task_idx_counts == 0] = 0
        else:
            task_weight_coefficients = np.fromiter(
                (config.__dict__["task_coef" + str(i)] for i in range(3)),
                dtype=np.float32
            )
        if config.exclude_task is not None:
            # in case the sequences corresponding to the task to be ignored have not been
            # excluded from the input dataset
            task_weight_coefficients[config.exclude_task] = 0
        model = KgiLMBert(
            model_config,
            from_pretrained=config.model_path if not config.from_scratch else None,
            num_labels_link_pred=num_labels_link_pred,
            task_weight_coefficients=task_weight_coefficients.tolist()
        )
    optimizer = AdamW(tuple(model.parameters()), lr=config.learning_rate)
    if config.from_cp:
        optimizer_state_dict = torch.load(os.path.join(config.model_path, "optimizer.bin"))
        optimizer.load_state_dict(optimizer_state_dict)

    now = datetime.now()
    output_subdir = f"{config.model_run_name}_{now.day}-{now.month}_{now.hour}-{now.minute}"
    output_fp = os.path.join(os.getenv("HOME"), "mbiolm-proj/kgi-runs", output_subdir)
    if not os.path.isdir(output_fp):
        os.mkdir(output_fp)
    with open(os.path.join(output_fp, "script_params.json"), "w", encoding=TEXT_ENC) as f_io:
        json.dump(config.as_dict(), f_io)

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
    logger.info("Effective Batch Size: %d", config.batch_size * accelerator.num_processes * config.grad_acc)
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
    logger.info("Accelerator state - %s", accelerator.state.__repr__()[:-1].replace("\n", "; "))
    logger.info("=== Starting Training Loop ===")

    def save_checkpoint():
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

    wandb_active = isinstance(config, wandb.sdk.wandb_config.Config)
    if wandb_active:
        wandb.watch(model, log_freq=config.wandb_log_freq)
        setup_metric_step_axes()
    eval_loss_list = []
    n_examples_train, n_examples_eval = 0, 0
    for epoch in range(config.epochs):
        model.train()
        if wandb_active:
            wandb.log({"epoch": epoch})
        for idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                n_examples_train += config.batch_size
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if wandb_active and idx % config.wandb_log_freq == 0:
                loss_ = loss.detach().item()
                metrics_to_log = {"train_step": n_examples_train, "train_loss": loss_}
                wandb.log(metrics_to_log)

        model.eval()
        batch_eval_loss = []
        for idx, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss_val = outputs.loss.detach().item()
            if np.isnan(loss_val):  # usually underflow afaik
                loss_val = 1e-6
            n_examples_eval += config.batch_size
            if wandb_active and idx % config.wandb_log_freq == 0:
                metrics_to_log = {"eval_step": n_examples_eval, "eval_loss": loss_val}
                wandb.log(metrics_to_log)
            batch_eval_loss.append(loss_val)
        epoch_eval_loss = sum(batch_eval_loss) / len(batch_eval_loss)
        eval_loss_list.append(epoch_eval_loss)

        logger.info(
            "\tFinished epoch %d - eval. loss = %d...",
            epoch, round(epoch_eval_loss, 3)
        )
        if (epoch + 1) % config.epoch_checkpoint_interval == 0 and epoch != config.epochs - 1:
            save_checkpoint()

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
        save_checkpoint()
        with open(
            os.path.join(output_fp, "eval_loss_by_epoch.json"), "w", encoding=TEXT_ENC
        ) as f_io:
            json.dump({"loss": eval_loss_list}, f_io)
    logger.info("Done!")


def main(args):
    # config setup
    with open(args.config_file, encoding=TEXT_ENC) as f_io:
        config = {**vars(args), **json.load(f_io)}

    if args.wandb_proj and args.wandb_entity:
        with wandb.init(project=args.wandb_proj, entity=args.wandb_entity, config=config):
            config = wandb.config
            logger.info("Launching training with W&B...")
            run_pipeline(config)
    else:
        logger.info("Launching training...")
        config = Bunch(**config)
        run_pipeline(config)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format=LOGFMT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)
    main(parse_arguments())
    