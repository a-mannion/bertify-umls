"""Helper functions and classes for processing the mixed-objective `BERTified` UMLS datasets
"""
import os
import re
import sys
import json
import warnings
from functools import partial
from collections import defaultdict
from string import punctuation
from random import sample
from pathlib import Path
from typing import Union, Optional, List, Tuple, Type, Dict, Callable

import torch
from torch.utils import data
from pandas import read_csv, DataFrame
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerFast
)

TEXT_ENC = sys.getdefaultencoding()

class Bunch:
    """Basic container that helps with training script readability"""
    _acceptable_types = float, int, bool, type(None), str, torch.Tensor

    def __init__(self, **kwargs):
        checked_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, self._acceptable_types)}
        self.__dict__.update(checked_kwargs)

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            return None

    def as_dict(self):
        return self.__dict__


class BatchEncodingDataset(data.Dataset):
    """Pytorch dataset that unwraps transformers batch encodings to be more easily accessible
    by data loaders"""

    def __init__(self, data_, return_tensors=True):
        super().__init__()
        if isinstance(data_, BatchEncoding):
            self._init_sequences(data_)
        elif isinstance(data_, (list, tuple)):
            for encoding in data_:
                if isinstance(encoding, BatchEncoding):
                    if hasattr(self, "_keys"):
                        for k in self._keys:
                            self.__dict__[k] += encoding[k]
                    else:
                        self._init_sequences(encoding)
        self.return_tensors = return_tensors

    def __len__(self):
        return len(getattr(self, self._keys[0]))

    def __getitem__(self, item):
        if self.return_tensors:
            return {key: torch.as_tensor(getattr(self, key)[item]) for key in self._keys}
        return {key: getattr(self, key)[item] for key in self._keys}

    def _init_sequences(self, batch_encoding):
        self._keys = tuple(batch_encoding.keys())
        self.__dict__.update(dict(batch_encoding.items()))


def load_kgi_tokenizer(
    name_or_path: Union[str, os.PathLike],
    add_bert_special_tokens: bool,
    load_via_hf: bool,
    relation_tokens: List[str],
    model_max_length: int,
    fpt_kwargs: Optional[Dict]=None
) -> PreTrainedTokenizerFast:
    bert_special_tokens = "SEP", "CLS", "UNK", "MASK", "PAD"
    tokenizer_special_token_kwargs = {
        t.lower() + "_token": "[" + t + "]" for t in bert_special_tokens
    } if add_bert_special_tokens else {}
    if load_via_hf:
        if fpt_kwargs:
            tokenizer_special_token_kwargs.update(fpt_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            model_max_length=model_max_length,
            **tokenizer_special_token_kwargs,
        )
    else:
        tokenizer = PreTrainedTokenizerFast(
            model_max_length=model_max_length,
            padding_side="right",
            truncation_side="right",
            tokenizer_file=name_or_path,
            **tokenizer_special_token_kwargs
        )
    if relation_tokens:
        n_tokens_to_add = len(relation_tokens) + 1
        n_tokens_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": relation_tokens + ["[HREL]"]}
        )
        assert n_tokens_added == n_tokens_to_add, \
            "failed to add all relation tokens"
    return tokenizer


def load_kgi_model_checkpoint(
    filepath: Union[str, os.PathLike],
    model_class: Type[PreTrainedModel],
    transformer_model_attr: str,
    # remove_key_prefix: bool=True,
    remove_modules_from_transformer: Optional[List[str]]=None,
    vocab_size: Optional[int]=None,
    model_init_kwargs: Optional[dict]=None,
) -> PreTrainedModel:
    with open(os.path.join(filepath, "config.json"), encoding=TEXT_ENC) as f_io:
        config_dict = json.load(f_io)
    model_type_name = config_dict.pop("_name_or_path")
    if model_init_kwargs:
        config_dict.update(model_init_kwargs)
    try:
        config_obj = AutoConfig.from_pretrained(model_type_name, **config_dict)
    except OSError:
        config_obj = AutoConfig.from_pretrained(
            os.path.join(os.getenv("HOME"), model_type_name),
            **config_dict
        )
    transformer = AutoModel.from_config(config_obj)
    torch_dict = torch.load(os.path.join(filepath, "pytorch_model.bin"))
    transformer_state_dict = {
        k.replace("transformer.", "", 1): v for k, v in torch_dict.items() \
            if k.startswith("transformer")
    }
    transformer.load_state_dict(transformer_state_dict)
    if remove_modules_from_transformer:
        for module_str in remove_modules_from_transformer:
            setattr(transformer, module_str, None)
    model = model_class.from_config(config_obj)
    setattr(model, transformer_model_attr, transformer)
    if vocab_size:
        model.resize_token_embeddings(vocab_size)
    return model


def prepare_mixed_dataset(
    kb_datadir: Union[str, os.PathLike],
    corpus_fp: Union[str, os.PathLike],
    tokenizer: Union[str, os.PathLike, PreTrainedTokenizerFast],
    load_tokenizer_via_hf: bool,
    model_max_length: int,
    n_text_docs: Optional[int]=None,
    train_set_frac: Optional[float]=None,
    add_bert_special_tokens: bool=True,
    shuffle: bool=False,
    tokenizer_fpt_kwargs: Optional[Dict]=None
) -> Union[
    Tuple[BatchEncodingDataset],
    Tuple[BatchEncodingDataset, BatchEncodingDataset],
    Tuple[BatchEncodingDataset, PreTrainedTokenizerFast],
    Tuple[BatchEncodingDataset, BatchEncodingDataset, PreTrainedTokenizerFast]
]:
    """Main KGI dataset generation function
    
    Parameters
    ----------
    kb_datadir:
        Directory containing the dataset (generated using `build_dataset.py`)
    corpus_fp:
        Path to the text file to use for MLM; if it's a directory, all .txt files in
        the directory will be loaded
    tokenizer:
        Tokenizer instance to use - can also be a path allowing transformers to load
        a tokenizer
    load_tokenizer_via_hf:
        In the casse where `tokenizer` is a string, indicate whether to use the
        transfomers `AutoTokenizer.from_pretrained` method to load it
    model_max_length:
        Maximal sequence length to parameterise the tokenizer - unused if `tokenizer`
        is passed as an already-instantiated tokenizer object
    n_text_docs:
        Number of documents to load from the `corpus_fp` file, which is assumed to be
        a text file containing documents separated by double line breaks
    train_set_frac:
        Proportion of the data to use as a training set (the rest will be used as an
        eval set to track the loss during training)
    add_bert_special_tokens:
        Whether to manually add padding, separation, cls, masking & padding tokens to
        the tokenizer vocabulary - usually not necessary when using a pre-trained
        tokenizer
    shuffle:
        Randomise the order of the training sequences before BERT-specific processing
    """
    kb_sequence_dataset = _build_kb_sequence_dataset(kb_datadir, shuffle=shuffle)
    if os.path.isfile(corpus_fp):
        sentences = _load_sentences(corpus_fp, n_text_docs)
    elif os.path.isdir(corpus_fp):
        sentences = []
        for txtfile in Path(corpus_fp).glob("*.txt"):
            sentences.extend(_load_sentences(txtfile, n_text_docs))
    else:
        raise ValueError(f"Not a valid path: {corpus_fp}")
    if tokenizer is None:
        return kb_sequence_dataset, sentences
    return_tokenizer = False
    if isinstance(tokenizer, str):
        return_tokenizer = True
        tokenizer = load_kgi_tokenizer(
            tokenizer,
            add_bert_special_tokens=add_bert_special_tokens,
            load_via_hf=load_tokenizer_via_hf,
            relation_tokens=kb_sequence_dataset["relation_tokens"],
            model_max_length=model_max_length,
            fpt_kwargs=tokenizer_fpt_kwargs
        )
    tokenize = partial(
        tokenizer,
        return_special_tokens_mask=True,
        padding=False,
        truncation=True
    )
    triple_encoding_entpred = tokenize(kb_sequence_dataset["triples_ep"])
    triple_encoding_clf = tokenize(kb_sequence_dataset["triples_clf"]["text"])
    triple_encoding_clf["labels"] = kb_sequence_dataset["triples_clf"]["labels"]
    path_encoding = tokenize(kb_sequence_dataset["paths"])
    corpus_encoding = tokenize(sentences)

    # indicators for sequence types: 0 for entity prediction triples, 1 for paths,
    # 2 for classification triples, and 3 for sentences
    # and also add a `labels` attribute to the paths and sentences so that they can go in a
    # BatchEncodingDataset with the triples; the actual training labels will be
    # added at collation
    split_dataset = list()
    for i, enc in enumerate((
        triple_encoding_entpred, path_encoding, triple_encoding_clf, corpus_encoding
    )):
        enc["task_type_index"] = (i * torch.ones(len(enc["input_ids"]))).tolist()
        if "labels" not in enc:
            enc["labels"] = [None for _ in range(len(enc["input_ids"]))]
        if train_set_frac:
            split_enc = defaultdict(dict)
            for k, v in enc.items():
                n_train = int(train_set_frac * len(v))
                train_data, eval_data = v[:n_train], v[n_train:]
                split_enc["train"][k] = train_data
                split_enc["eval"][k] = eval_data
            split_dataset.append(split_enc)

    if split_dataset:
        train_dataset_input = [BatchEncoding(enc["train"]) for enc in split_dataset]
        eval_dataset_input = [BatchEncoding(enc["eval"]) for enc in split_dataset]
        dataset = BatchEncodingDataset(train_dataset_input, return_tensors=False), \
            BatchEncodingDataset(eval_dataset_input, return_tensors=False)
    else:
        dataset_input = triple_encoding_entpred, path_encoding, triple_encoding_clf, corpus_encoding
        dataset = BatchEncodingDataset(dataset_input, return_tensors=False),
    if return_tokenizer:
        return *dataset, tokenizer
    return dataset


def get_relation_labels(
    tokenizer: PreTrainedTokenizerFast,
    relation_token_ids: Optional[List[int]]=None,
    exclude_relation_types: Optional[Tuple[str]]=("[SY]",)
) -> Dict[str, int]:
    """Helper function for generating training labels for the link prediction task"""
    if not relation_token_ids:
        relation_token_ids = tokenizer.additional_special_tokens_ids[:-1]  # exclude [HREL]
    if exclude_relation_types:
        token_ids_to_exclude = tokenizer.convert_tokens_to_ids(exclude_relation_types)
        relation_token_ids = [
            rt_id for rt_id in relation_token_ids if rt_id not in token_ids_to_exclude
        ]
    # in order for a (n_relations)-dimensional classifier layer to work, we need to map the special
    # tokens to integers in [0, n_relations]
    return {t: i for i, t in enumerate(relation_token_ids)}


def mixed_collate_fn(
    batch_examples: List[BatchEncoding],
    tokenizer: PreTrainedTokenizerFast,
    rel_token_ids2labels: Optional[dict]=None,
    return_enc_dicts: bool=False,
    exclude_relation_types: Optional[Tuple[str]]=None
) -> Union[Tuple[dict], BatchEncoding]:
    """Data collation function to pass to the DataLoader for UMLS-KGI training. This implements
    `smart batching`, whereby sequences are padded to the maximal length within individual batches
    instead of fixing a maximum length to apply to the whole dataset; this can result in significant
    speedups in situations such as these where there's a lot of variation in sequence length"""
    
    # padding
    pad_id_mapping = {
        "attention_mask": 0,
        "token_type_ids": tokenizer.pad_token_type_id,
        "special_tokens_mask": 1,
        "input_ids": tokenizer.pad_token_id
    }
    max_len = max(len(b["input_ids"]) for b in batch_examples)
    for batch_ex in batch_examples:
        for k, data_val in batch_ex.items():
            if isinstance(data_val, list) and k in pad_id_mapping:
                diff = max_len - len(data_val)
                padded_seq = data_val + [pad_id_mapping[k]] * diff
                batch_ex[k] = torch.tensor(padded_seq, dtype=torch.long)    

    # split into tasks
    entpred_triples = [b for b in batch_examples if b["task_type_index"] == 0]
    paths = [b for b in batch_examples if b["task_type_index"] == 1]
    clf_triples = [b for b in batch_examples if b["task_type_index"] == 2]
    sentences = [b for b in batch_examples if b["task_type_index"] == 3]
    hrel_token_id = tokenizer.additional_special_tokens_ids[-1]
    relation_token_ids = tokenizer.additional_special_tokens_ids[:-1]
    if rel_token_ids2labels is None:
        rel_token_ids2labels = get_relation_labels(
            tokenizer, relation_token_ids, exclude_relation_types
        )

    # encode/collate the different sequence types separately
    model_input_names = ["input_ids", "attention_mask", "labels", "task_type_index"]
    if entpred_triples:
        triple_encodings_ep = _entity_prediction_collation(
            entpred_triples,
            mask_token_id=tokenizer.mask_token_id,
            relation_token_ids=relation_token_ids,
            model_input_names=model_input_names,
            return_dict=True
        )
        enc_dicts = triple_encodings_ep,
    else:
        enc_dicts = ()
    if paths:
        path_encodings = _link_prediction_collation(
            paths,
            mask_token_id=hrel_token_id,
            relation_token_ids=relation_token_ids,
            model_input_names=model_input_names,
            rel_token_ids2labels=rel_token_ids2labels,
            return_dict=True
        )
        enc_dicts = *enc_dicts, path_encodings
    if clf_triples:
        clf_triple_data = defaultdict(list)
        for dict_ in clf_triples:
            for k in model_input_names:
                val = torch.as_tensor(dict_[k]) if isinstance(dict_[k], list) else dict_[k]
                clf_triple_data[k].append(val)
        clf_triples_encoding = BatchEncoding(clf_triple_data)
        enc_dicts = *enc_dicts, clf_triples_encoding
    if sentences:
        input_shape = torch.Size((len(sentences), max_len))
        mask_idx, mask_token_idx, random_idx = _make_mlm_masks(
            input_shape=input_shape,
            prob=.15,
            special_tokens_mask=torch.stack([s["special_tokens_mask"] for s in sentences])
        )
        sentence_ids = torch.stack([s["input_ids"] for s in sentences])
        labels_tensor = sentence_ids.clone()
        labels_tensor[~mask_idx] = -100
        labels = [labels_tensor[i] for i in range(labels_tensor.shape[0])]
        sentence_encodings = {}
        sentence_encodings["labels"] = labels
        sentence_ids[mask_token_idx] = tokenizer.mask_token_id
        sentence_ids[random_idx] = torch.randint(len(tokenizer), (random_idx.sum(),))
        sentence_encodings["input_ids"] = sentence_ids
        sentence_encodings["attention_mask"] = torch.stack([s["attention_mask"] for s in sentences])
        sentence_encodings["task_type_index"] = [s["task_type_index"] for s in sentences]
        enc_dicts = *enc_dicts, sentence_encodings

    if return_enc_dicts:
        return enc_dicts

    # stack all the separate task tensors back together
    output_dict = {}
    for i, k in enumerate(model_input_names):
        output_value = []
        for enc_dict in enc_dicts:
            if isinstance(enc_dict[k], torch.Tensor):
                output_value.append(enc_dict[k])
            elif isinstance(enc_dict[k], list):
                output_value.extend(enc_dict[k])
        if k != "labels":  # labels can have varying dimensions so need to be kept as a list
            tensor_elems = []
            for elem in output_value:
                if isinstance(elem, torch.Tensor):
                    if len(elem.shape) == 1:
                        tensor_elems.append(elem.unsqueeze(0))
                    else:
                        tensor_elems.append(elem)
                elif isinstance(elem, (int, float)):
                    tensor_elems.append(torch.as_tensor(elem).reshape((1, 1)))
                else:
                    warnings.warn(
                        f"Collation: found {k} value with unexpected type {type(elem)}"
                        "; this will probably cause training errors",
                        RuntimeWarning
                    )
                    tensor_elems.append(elem)
            try:
                output_value = torch.cat(tensor_elems).squeeze()
            except RuntimeError as rterr:
                def _showtensortypes(tensor_elems):
                    vals, txt = [], []
                    for t_elem in tensor_elems:
                        if isinstance(t_elem, torch.Tensor):
                            vals.append(t_elem.dtype)
                            txt.append("tensor: ")
                        else:
                            vals.append(type(t_elem))
                            txt.append("not tensor, ")
                    return ", ".join(t + str(v) for t, v in zip(txt, vals))
                warnings.warn(
                    f"Collation: failed to concatenate {k} tensors: " + \
                    _showtensortypes(tensor_elems) + \
                    f", because:\n{rterr}"
                )
        if not i:  # first iteration
            l = len(output_value)
        else:
            if l != len(output_value):
                warnings.warn(
                    f"Collation: came across inconsistent batch element length {l} for {k}:\n"
                    f"{', '.join(k + ': ' + str(len(v)) for k, v in output_value.items())}"
                )
        output_dict[k] = output_value
    return BatchEncoding(output_dict)


def _build_kb_sequence_dataset(
    dir_: Union[str, os.PathLike],
    shuffle: bool=False
) -> Dict[str, Union[Dict[str, List[str]], List[str]]]:
    punc = re.sub(r"'|\[|\]", "", punctuation)
    clean_f = partial(_entity_text_prep, trans=str.maketrans(punc, " " * len(punc)))
    with open(os.path.join(dir_, "paths.json"), encoding=TEXT_ENC) as f_io:
        path_dataset = json.load(f_io)
    path_sequences = _make_path_sequences(path_dataset, clean_f)
    kwargs = {"sep": "\t", "engine": "pyarrow", "dtype_backend": "pyarrow"}
    entpred_dataset = read_csv(os.path.join(dir_, "triples.tsv"), **kwargs)
    relation_tokens = entpred_dataset.REL.drop_duplicates().apply(lambda x: f"[{x}]").to_list()
    ep_sequences = _make_triple_sequence_list(entpred_dataset, clean_f)
    triclf_dataset = read_csv(os.path.join(dir_, "triple_clf.tsv"), **kwargs)
    if shuffle:
        ep_sequences = sample(ep_sequences, len(ep_sequences))
        path_sequences = sample(path_sequences, len(path_sequences))
        triclf_dataset = triclf_dataset.sample(frac=1.)
    clf_sequences = {
        "text": _make_triple_sequence_list(triclf_dataset, clean_f),
        "labels": triclf_dataset.clf_label.tolist()
    }
    output_data = {
        "triples_ep": ep_sequences,
        "triples_clf": clf_sequences,
        "paths": path_sequences,
        "relation_tokens": relation_tokens
    }
    return output_data


def _load_sentences(data_fp: Union[str, os.PathLike], n_docs: int) -> List[str]:
    with open(data_fp, encoding=TEXT_ENC) as f_io:
        if n_docs is None:
            text = [doc for doc in f_io.read().split("\n")]
        else:
            text = []
            for i, line in enumerate(f_io):
                text.append(line.replace("\n", ""))
                if i == n_docs:
                    break
    return text


def _entity_prediction_collation(
    batch_examples: List[dict],
    mask_token_id: int,
    relation_token_ids: List[int],
    model_input_names: List[str],
    return_dict: bool=False
) -> Union[Dict[str, List[torch.Tensor]], BatchEncoding]:
    masked_encoding = defaultdict(list)
    for dict_ in batch_examples:
        labels = dict_["input_ids"].clone()
        for id_ in relation_token_ids:
            try:
                rel_idx = (dict_["input_ids"] == id_).nonzero().item()
            except (ValueError, RuntimeError):
                continue
            break
        try:
            tail_entity_mask = (torch.arange(len(dict_["input_ids"])) > rel_idx) \
                & ~dict_["special_tokens_mask"].bool()
        except UnboundLocalError:
            # no relation tokens in the sequence
            tail_entity_mask = ~dict_["special_tokens_mask"].bool()
        dict_["input_ids"].masked_fill_(tail_entity_mask, mask_token_id)
        dict_["labels"] = labels.masked_fill(~tail_entity_mask, -100)
        for k in model_input_names:
            val = torch.as_tensor(dict_[k]) if isinstance(dict_[k], list) else dict_[k]
            masked_encoding[k].append(val)
    if return_dict:
        return masked_encoding
    return BatchEncoding(masked_encoding)


def _link_prediction_collation(
    batch_examples: List[dict],
    mask_token_id: int,
    relation_token_ids: List[int],
    model_input_names: List[str],
    rel_token_ids2labels: Dict[int, int],
    return_dict: bool=False
) -> Union[Dict[str, List[torch.Tensor]], BatchEncoding]:
    masked_encoding = defaultdict(list)
    for dict_ in batch_examples:
        labels = dict_["input_ids"].clone()
        rel_mask = torch.isin(dict_["input_ids"], torch.tensor(relation_token_ids))
        dict_["input_ids"].masked_fill_(rel_mask, mask_token_id)
        dict_["labels"] = labels.masked_fill(~rel_mask, -100)
        dict_["labels"].apply_(
            lambda x: rel_token_ids2labels[x] if x in rel_token_ids2labels else x
        )
        for k in model_input_names:
            val = torch.as_tensor(dict_[k]) if isinstance(dict_[k], list) else dict_[k]
            masked_encoding[k].append(val)
    if return_dict:
        return masked_encoding
    return BatchEncoding(masked_encoding)


def _make_mlm_masks(
    input_shape: Tuple[int],
    prob: float,
    special_tokens_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.full(input_shape, prob)
    if special_tokens_mask.dtype != torch.bool:
        special_tokens_mask = special_tokens_mask.bool()
    probs.masked_fill_(special_tokens_mask, value=0.)
    mask_idx = probs.bernoulli().bool()
    mask_token_idx = torch.bernoulli(torch.full(input_shape, .8)).bool() & mask_idx
    random_idx = torch.bernoulli(torch.full(input_shape, .1)).bool() & mask_idx & ~mask_token_idx
    return mask_idx, mask_token_idx, random_idx


def _entity_text_prep(text: str, trans: dict) -> str:
    return re.sub(r"\s+", " ", text.translate(trans)).strip()


def _make_path_sequences(dict_: dict, clean_f: Callable) -> List[str]:
    sequences = []
    for path in dict_.values():
        seq = " ".join(list(clean_f(v) if k != "REL" else f"[{v}]" for k, v in path["t0"].items()))
        for k, data_val in path.items():
            if k == "t0":
                continue
            seq += " [" + data_val["REL"] + "] " + clean_f(data_val["STR"])
        sequences.append(seq)
    return sequences


def _make_triple_sequence_list(dataset: DataFrame, clean_f: Callable) -> List[str]:
    res = dataset.STR2.apply(clean_f) + dataset.REL.apply(lambda x: f" [{x}] ") + \
        dataset.STR1.apply(clean_f)
    return res.to_list()
