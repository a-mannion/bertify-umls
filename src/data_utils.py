import os
import re
import sys
import json
from functools import partial
from collections import defaultdict
from string import punctuation
from random import sample

import torch
from torch.utils import data
from pandas import read_csv
from numpy.random import Generator as RNG
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast

TEXT_ENC = sys.getdefaultencoding()

class Bunch:

    def __init__(self, dict_=None, **kwargs):
        if not isinstance(dict_, dict):
            dict_ = None
        self.__dict__.update({**dict_, **kwargs})

    def as_dict(self):
        return self.__dict__


class BatchEncodingDataset(data.Dataset):

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


class BatchShuffleIterator(data.dataloader._MultiProcessingDataLoaderIter):

    def __next__(self):
        batch = super().__next__()
        if isinstance(self.loader, BatchShuffleDataLoader):
            if self.loader.shuffle:
                # shuffle the sequences within the batch
                RNG.shuffle(batch, axis=1)
        return batch


class BatchShuffleDataLoader(data.DataLoader):

    def __init__(self, dataset, shuffle, **kwargs):
        # assume that `dataset` is an ordered BatchEncodingDataset,
        # and that `kwargs` contains a `collate_fn` parameter set to
        # some parameterised instance of `smart_batch_mlm_collate_fn`
        super().__init__(dataset, **kwargs)
        self.shuffle = shuffle

    def __iter__(self):
        self.batch_sampler.shuffle()
        return BatchShuffleIterator(self)


def prepare_mixed_dataset(
    kb_datadir,
    corpus_fp,
    tokenizer,
    load_tokenizer_via_hf,
    model_max_length,
    n_text_docs=None,
    train_set_frac=None,
    add_bert_special_tokens=True,
    shuffle=False
):
    kb_sequence_dataset = _build_kb_sequence_dataset(kb_datadir, shuffle=shuffle)
    sentences = _load_sentences(corpus_fp, n_text_docs)
    if tokenizer is None:
        return kb_sequence_dataset, sentences
    return_tokenizer = False
    if isinstance(tokenizer, str):
        bert_special_tokens = "SEP", "CLS", "UNK", "MASK", "PAD"
        tokenizer_special_token_kwargs = {
            t.lower() + "_token": "[" + t + "]" for t in bert_special_tokens
        } if add_bert_special_tokens else {}
        return_tokenizer = True
        if load_tokenizer_via_hf:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer,
                model_max_length=model_max_length,
                **tokenizer_special_token_kwargs
            )
        else:
            tokenizer = PreTrainedTokenizerFast(
                model_max_length=model_max_length,
                padding_side="right",
                truncation_side="right",
                tokenizer_file=tokenizer,
                **tokenizer_special_token_kwargs
            )
        n_tokens_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": kb_sequence_dataset["relation_tokens"] + ["[HREL]"]}
        )
        assert n_tokens_added == len(kb_sequence_dataset["relation_tokens"]) + 1, \
            "failed to add all relation tokens"
    tokenize = partial(tokenizer, return_special_tokens_mask=True, padding=False, truncation=True)
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


def get_relation_labels(tokenizer, relation_token_ids=None, exclude_relation_types=("[SY]",)):
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
    batch_examples,
    tokenizer,
    rel_token_ids2labels=None,
    return_enc_dicts=False,
    exclude_relation_types=None
):
    # padding
    pad_id_mapping = dict(
        attention_mask=0,
        token_type_ids=tokenizer.pad_token_type_id,
        special_tokens_mask=1,
        input_ids=tokenizer.pad_token_id
    )
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
                clf_triple_data[k].append(torch.as_tensor(dict_[k]))
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
        labels = sentence_ids.clone()
        labels[~mask_idx] = -100
        sentence_encodings = dict()
        sentence_encodings["labels"] = labels
        sentence_ids[mask_token_idx] = tokenizer.mask_token_id
        sentence_ids[random_idx] = torch.randint(tokenizer.vocab_size, (random_idx.sum(),))
        sentence_encodings["input_ids"] = sentence_ids
        sentence_encodings["attention_mask"] = torch.stack([s["attention_mask"] for s in sentences])
        sentence_encodings["task_type_index"] = torch.tensor(
            [s["task_type_index"] for s in sentences]
        )
        enc_dicts = *enc_dicts, sentence_encodings

    if return_enc_dicts:
        return enc_dicts

    # stack all the separate task tensors back together
    output_dict = {}
    for k in model_input_names:
        output_value = []
        for enc_dict in enc_dicts:
            if isinstance(enc_dict[k], torch.Tensor):
                val = enc_dict[k]
            elif isinstance(enc_dict[k], list):
                if isinstance(enc_dict[k][0], torch.Tensor):
                    val = torch.stack(enc_dict[k])
                elif isinstance(enc_dict[k][0], (int, float)):
                    val = torch.tensor(enc_dict[k])
            output_value.append(val)
        # labels aren't going to all have the same dims
        output_dict[k] = output_value if k == "labels" else torch.cat(output_value)
    return BatchEncoding(output_dict)


def _build_kb_sequence_dataset(dir_, shuffle=False):
    punc = re.sub(r"'|\[|\]", "", punctuation)
    clean_f = partial(_entity_text_prep, trans=str.maketrans(punc, " " * len(punc)))
    with open(os.path.join(dir_, "paths.json"), encoding=TEXT_ENC) as f_io:
        path_dataset = json.load(f_io)
    path_sequences = _make_path_sequences(path_dataset, clean_f)
    kwargs = dict(sep="\t", engine="pyarrow", dtype_backend="pyarrow")
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


def _load_sentences(data_fp, n_docs):
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


def _entity_prediction_collation(batch_examples, mask_token_id, relation_token_ids, model_input_names, return_dict=False):
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
            masked_encoding[k].append(torch.as_tensor(dict_[k]))
    if return_dict:
        return masked_encoding
    return BatchEncoding(masked_encoding)


def _link_prediction_collation(batch_examples, mask_token_id, relation_token_ids, model_input_names, rel_token_ids2labels, return_dict=False):
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
            masked_encoding[k].append(dict_[k])
    if return_dict:
        return masked_encoding
    return BatchEncoding(masked_encoding)


def _make_mlm_masks(input_shape, prob, special_tokens_mask):
    probs = torch.full(input_shape, prob)
    probs.masked_fill_(special_tokens_mask, value=0.)
    mask_idx = probs.bernoulli().bool()
    mask_token_idx = torch.bernoulli(torch.full(input_shape, .8)).bool() & mask_idx
    random_idx = torch.bernoulli(torch.full(input_shape, .1)).bool() & mask_idx & ~mask_token_idx
    return mask_idx, mask_token_idx, random_idx


def _entity_text_prep(text, trans):
    return re.sub(r"\s+", " ", text.translate(trans)).strip()


def _make_path_sequences(dict_, clean_f):
    sequences = []
    for path in dict_.values():
        seq = " ".join(list(clean_f(v) if k != "REL" else f"[{v}]" for k, v in path["t0"].items()))
        for k, data_val in path.items():
            if k == "t0":
                continue
            seq += " [" + data_val["REL"] + "] " + clean_f(data_val["STR"])
        sequences.append(seq)
    return sequences


def _make_triple_sequence_list(dataset, clean_f):
    res = dataset.STR2.apply(clean_f) + dataset.REL.apply(lambda x: f" [{x}] ") + \
        dataset.STR1.apply(clean_f)
    return res.to_list()
