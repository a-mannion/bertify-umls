"""Model classes for UMLS-KGI-BERT training
"""
import os
import sys
import json

from transformers.activations import gelu
import torch
from transformers import (
    AutoModel,
    PretrainedConfig,
    PreTrainedModel
)

from data_utils import Bunch

TEXT_ENC = sys.getdefaultencoding()


class LMHead(torch.nn.Module):
    """MLM module copied from CamemBERT/RoBERTa"""

    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.dim, config.dim)
        self.layer_norm = torch.nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.decoder = torch.nn.Linear(config.dim, config.vocab_size)
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class ClassificationHead(torch.nn.Module):
    """Standard linear classifier"""

    def __init__(self, dim, output_size, dropout_size):
        super().__init__()
        self.pre_classifier = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout_size)
        self.classifier = torch.nn.Linear(dim, output_size)

    def forward(self, features):
        x = self.pre_classifier(features)
        x = gelu(x)
        x = self.classifier(x)
        return x


class KgiLMBert(PreTrainedModel):
    """BERT model wrapper that implements a joint objective function composed of knowledge graph
    triple classification, link prediction, and entity completion alongside masked language
    modelling
    """

    def __init__(
        self,
        config,
        from_pretrained=None,
        state_dict=None,
        num_labels_link_pred=6,
        task_weight_coefficients=None,
        triple_clf_dropout=.1,
        link_pred_dropout=.1
    ):
        if isinstance(config, dict):
            config = PretrainedConfig.from_dict(config)
        elif not isinstance(config, PretrainedConfig):
            raise RuntimeError(f"Invalid type for `config` argument: {type(config)}")
        super().__init__(config)
        if not hasattr(config, "dim"):
            dim = config.hidden_size if config.hidden_size else 2048
            config.__dict__.update({"dim": dim})
        if not hasattr(config, "layer_norm_eps"):
            config.__dict__.update({"layer_norm_eps": 1e-12})
        self.triple_clf_dropout = triple_clf_dropout
        self.link_pred_dropout = link_pred_dropout
        self.num_labels_link_pred = num_labels_link_pred
        self.task_weight_coefficients = [0.5] * 3 if task_weight_coefficients is None \
            else task_weight_coefficients
        if from_pretrained:
            self.transformer = AutoModel.from_pretrained(
                from_pretrained,
                state_dict=state_dict,
                ignore_mismatched_sizes=True
            )
            # embedding input dim will be different because of the relation tokens
            self.transformer.resize_token_embeddings(config.vocab_size)
        else:
            self.transformer = AutoModel.from_config(config)
        self.config = config
        self.loss_fct = torch.nn.CrossEntropyLoss()
        clf_classes, module_list, nonzero_task_weight_coefficients = [], [], []

        ### MLM
        self.lm_head = LMHead(config)

        if self.task_weight_coefficients[0] > 0:
            clf_classes.append(config.vocab_size)
            module_list.append("lm_head")
            nonzero_task_weight_coefficients.append(self.task_weight_coefficients[0])

        ### Link prediction (token classification)
        if self.task_weight_coefficients[1] > 0:
            self.link_classifier = ClassificationHead(
                config.dim,
                self.num_labels_link_pred,
                self.link_pred_dropout
            )
            clf_classes.append(self.num_labels_link_pred)
            module_list.append("link_classifier")
            nonzero_task_weight_coefficients.append(self.task_weight_coefficients[1])
        else:
            self.num_labels_link_pred = self.link_classifier = None

        ### Triple (Sequence) Classification
        if self.task_weight_coefficients[2] > 0:
            self.num_labels_triple_clf = 2
            self.triple_classifier = ClassificationHead(
                config.dim,
                self.num_labels_triple_clf,
                self.triple_clf_dropout
            )
            clf_classes.append(self.num_labels_triple_clf)
            module_list.append("triple_classifier")
            nonzero_task_weight_coefficients.append(self.task_weight_coefficients[2])
        else:
            self.num_labels_triple_clf = self.triple_classifier = None

        # MLM gets task index 3
        clf_classes.append(config.vocab_size)
        module_list.append("lm_head")

        self.n_tasks = len(nonzero_task_weight_coefficients) + 1
        self._clf_classes = clf_classes
        self._module_list = module_list
        self._nonzero_task_weight_coefficients = nonzero_task_weight_coefficients

        self.kgi_specific_config = {
            k: getattr(self, k) \
                for k in (
                    "num_labels_link_pred", "link_pred_dropout",
                    "num_labels_triple_clf", "triple_clf_dropout",
                    "task_weight_coefficients", "n_tasks", "_clf_classes",
                    "_module_list", "_nonzero_task_weight_coefficients"
                )
        }

        self.post_init()

    @classmethod
    def from_pretrained(cls, checkpoint):
        base_model = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
        with open(os.path.join(checkpoint, "config.json"), encoding=TEXT_ENC) as f_io:
            transformer_config = json.load(f_io)
        with open(os.path.join(checkpoint, "kgi_specific_config.json"), encoding=TEXT_ENC) as f_io:
            kgi_specific_config = json.load(f_io)

        transformer_state_dict = {
            k[12:]: v for k, v in base_model.items() if k.startswith("transformer")
        }
        self_ = KgiLMBert(
            transformer_config,
            from_pretrained=checkpoint,
            state_dict=transformer_state_dict,
            num_labels_link_pred=kgi_specific_config["num_labels_link_pred"],
            task_weight_coefficients=kgi_specific_config["task_weight_coefficients"],
            triple_clf_dropout=kgi_specific_config["triple_clf_dropout"],
            link_pred_dropout=kgi_specific_config["link_pred_dropout"]
        )
        self_.kgi_specific_config = kgi_specific_config
        return self_

    def forward(self, input_ids, attention_mask, labels, task_type_index):
        outputs = self.transformer(input_ids, attention_mask, output_hidden_states=False)
        sequence_output = outputs[0]
        task_types_in_batch = torch.unique(task_type_index).int()
        losses = torch.stack([
            self._get_loss(
                sequence_output,
                batch_labels=labels,
                task_type_index_batch=task_type_index,
                task_type_index_ref=task_type_index_ref.item()
            ) for task_type_index_ref in task_types_in_batch
        ])
        coefs = torch.tensor([*self._nonzero_task_weight_coefficients, 1]).to(losses.device)
        coefs = coefs[task_types_in_batch]  # remove coefficients for tasks that don't come up in this batch
        loss = (coefs * losses).sum()
        return Bunch(loss=loss)

    def _get_loss(
        self,
        sequence_output,
        batch_labels,
        task_type_index_batch,
        task_type_index_ref
    ):
        tti_idx, = torch.where(task_type_index_batch == task_type_index_ref)
        n_classes = self._clf_classes[task_type_index_ref]
        module_name = self._module_list[task_type_index_ref]
        task_specific_output = sequence_output[tti_idx]
        task_label_list = [batch_labels[i] for i in tti_idx]
        if task_type_index_ref == 2:
            # sequence classification: only use [CLS]
            task_specific_output = task_specific_output[:,0,:]
            # integer labels - convert list to tensor
            labels = torch.tensor(task_label_list)
        else:
            # tensor labels - stack list
            labels = torch.stack(task_label_list)
        labels = labels.to(sequence_output.device)
        labelmax = labels.max()
        label_mask_bool = labels != -100
        labelmin = labels[label_mask_bool].min().item() if label_mask_bool.sum().item() else 0
        assert labelmin >= 0 and labelmax < n_classes, \
            f"Task {task_type_index_ref}: expected labels in range (0,{n_classes - 1}) \
                but got ({labelmin},{labelmax})"
        module = getattr(self, module_name)
        prediction_output = module(task_specific_output).view(-1, n_classes)
        loss = self.loss_fct(prediction_output, labels.view(-1))
        return loss
