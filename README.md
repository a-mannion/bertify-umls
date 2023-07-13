# bertify-umls
Toolkit for jointly training BERT encoder models on the UMLS metathesaurus knowledge graph as well as free-text corpora using simple graph-based reasoning tasks alongside masked-language modelling.

The code in this repository accompanies the paper "UMLS-KGI-BERT: Data-Centric Knowledge Integration in Transformers for Biomedical Entity Recognition", published at the [ClinicalNLP workshop](https://clinical-nlp.github.io/2023/program.html) at ACL 2023.

- To build your own version of the UMLS-KGI datasets, use the `src/build_dataset.py` script.
- To train your own version of the UMLS-KGI-BERT model, either from-scratch or from a pre-trained checkpoint, use `training_scripts/train.py`
- To evaluate on a token classification task, use `training_scripts/token_classification.py` 

Details on how to parameterise the scripts can be obtained by passing the `-h` flag.
