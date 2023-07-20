# bertify-umls
Toolkit for jointly training BERT encoder models on the UMLS metathesaurus knowledge graph as well as free-text corpora using simple graph-based reasoning tasks alongside masked-language modelling.

The code in this repository accompanies the paper "UMLS-KGI-BERT: Data-Centric Knowledge Integration in Transformers for Biomedical Entity Recognition", published at the [ClinicalNLP workshop](https://clinical-nlp.github.io/2023/program.html) at ACL 2023.

- To build your own version of the UMLS-KGI datasets, use the `src/build_dataset.py` script.
- To train your own version of the UMLS-KGI-BERT model, either from-scratch or from a pre-trained checkpoint, use `training_scripts/train.py`
- To evaluate on a token classification task, use `training_scripts/token_classification.py` 

Details on how to parameterise the scripts can be obtained by running them with the `-h` flag.

Paper citation:
```@inproceedings{mannion-etal-2023-umls,
    title = "{UMLS}-{KGI}-{BERT}: Data-Centric Knowledge Integration in Transformers for Biomedical Entity Recognition",
    author = "Mannion, Aidan  and
      Schwab, Didier  and
      Goeuriot, Lorraine",
    booktitle = "Proceedings of the 5th Clinical Natural Language Processing Workshop",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.clinicalnlp-1.35",
    pages = "312--322",
    abstract = "Pre-trained transformer language models (LMs) have in recent years become the dominant paradigm in applied NLP. These models have achieved state-of-the-art performance on tasks such as information extraction, question answering, sentiment analysis, document classification and many others. In the biomedical domain, significant progress has been made in adapting this paradigm to NLP tasks that require the integration of domain-specific knowledge as well as statistical modelling of language. In particular, research in this area has focused on the question of how best to construct LMs that take into account not only the patterns of token distribution in medical text, but also the wealth of structured information contained in terminology resources such as the UMLS. This work contributes a data-centric paradigm for enriching the language representations of biomedical transformer-encoder LMs by extracting text sequences from the UMLS.This allows for graph-based learning objectives to be combined with masked-language pre-training. Preliminary results from experiments in the extension of pre-trained LMs as well as training from scratch show that this framework improves downstream performance on multiple biomedical and clinical Named Entity Recognition (NER) tasks. All pre-trained models, data processing pipelines and evaluation scripts will be made publicly available.",
}
```
