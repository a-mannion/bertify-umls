"""BERTification of the UMLS metathesaurus knowledge graph triples, i.e. construction of training
data for the UMLS-KGI pre-training task
There are three main parts to this:
    1. Creating the base tables - from the standard UMLS download, create three TSV files - `concepts_ref`
        for the preferred terms of each concept to be used (this one will have exactly one row per concept),
        `concepts` for other terms (non-unique with respect to CUIs), and `relations` for the relationships
        that exist among the sampled concepts. This step only needs to be run once, then each time you want
        to create a new triplet dataset just provide the directory containing these TSV files using the 
        `--load_base_tables` argument.
    2. Generating the triplet dataset - sampling from the base tables, create a set of triplets
    3. Generating training datasets - sampling in turn from the triplet dataset created in step 2, create
        text sequences that can be tokenized for UMLS-KGI training
"""
import os
import sys
import json
import logging
import warnings
from argparse import ArgumentParser, Namespace
from typing import Union, Dict, List, Tuple, Optional

import pandas as pd


# command line arguments
UMLS_DIR_HELP = """Path to the directory containing the base UMLS dataset (should end with
`2022AB/META`)"""
SRDEF_PATH_HELP = """Path to the basic information file for semantic types and relations (SRDEF
from the UMLS semantic network)"""
SG_PATH_HELP = """Path to the semantic groups text file"""
WRITE_PATH_HELP = """Directory in which to write the output datasets"""
LANG_HELP = """Filter the dataset by language if desired. Must be specified in the format
used by the LAT column in the concepts file, i.e. ENG, SPA, etc"""
LBT_HELP = """Path to a directory containing already-preprocessed tables of concepts and relations
- if not provided, will load data from `umls_dir` and create new base tables in `write_path`"""
NSAMPLES_HELP = """Number of examples in each training dataset - the final size of the triplet
classification dataset will be 2x this"""
NSAMPLESBASE_HELP = """Number of triplets to create from the base data tables - if not provided
will sample from all of the data"""
MAXPATHLEN_HELP = """Cutoff length for the random walks across the graph to create link prediction
sequences"""
LANGSTRATTYPE_HELP = """Type of stratification for use in multilingual datasets. Can take values
`eq` for the same number of samples from each language or `rel` for proportional sampling;
defaults to `none`. Note that this does not guarantee that the final outputs will conform
exactly to the chosen proportions when further random sampling is done at the level of semantic
groups."""


# Metathesaurus data fields
SRDEF_COLNAMES = "RT", "UI", "STY_or_RL", "STN_or_RTN", "DEF" \
    "EX", "UN", "NH", "ABR", "RIN"
SEMANTIC_GROUPS_COLNAMES = "Abbrev", "Name", "TUI", "TypeName"
MRSTY_COLNAMES = "CUI", "TUI", "STN", "STY", "ATUI", "CVF"
MRCONSO_COLNAMES = "CUI", "LAT", "TS", "LUI", "STT", "SUI", \
    "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", \
    "CODE", "STR", "SRL", "SUPPRESS", "CVF"
MRREL_COLNAMES = "CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", \
    "STYPE2", "RELA", "RUI", "SRUI", "SAB", "SL", "RG", "DIR", \
    "SUPPRESS", "CVF"
INCLUDE_GROUPS = "CHEM", "DISO", "ANAT", "PROC", "CONC", "DEVI", \
    "PHEN", "PHYS"
EXCLUDE_SEMANTIC_TYPES = (
    "Plant", "Fungus", "Animal", "Vertebrate",
    "Amphibian", "Bird", "Fish", "Reptile", "Mammal", "Event",
    "Activity", "Social Behaviour", "Daily or Recreational Activity",
    "Occupational Activity", "Governmental or Regulatory Activity",
    "Educational Activity", "Machine Activity", "Conceptual Entity",
    "Geographic Area", "Regulation or Law", "Organization",
    "Professional Society", "Self-help or Relief Organization",
    "Professional or Occupational Group", "Family Group",
    "Chemical Viewed Structurally", "Intellectual Product",
    "Language", "Eukaryote"
)


def parse_arguments() -> Namespace:
    """Command line parser"""
    parser = ArgumentParser()
    parser.add_argument("--umls_dir", type=str, help=UMLS_DIR_HELP)
    parser.add_argument("--srdef", type=str, help=SRDEF_PATH_HELP)
    parser.add_argument("--sg", type=str, help=SG_PATH_HELP)
    parser.add_argument("--writepath", type=str, help=WRITE_PATH_HELP, default=os.getenv("HOME"))
    parser.add_argument("--lang", type=str, nargs="+", default="ENG", help=LANG_HELP)
    parser.add_argument("--load_base_tables", type=str, help=LBT_HELP)
    parser.add_argument("--n_samples", type=int, default=10000, help=NSAMPLES_HELP)
    parser.add_argument("--n_samples_base", type=int, help=NSAMPLESBASE_HELP)
    parser.add_argument("--max_path_len", type=int, default=10, help=MAXPATHLEN_HELP)
    parser.add_argument(
        "--lang_strat_type",
        type=str,
        choices={"none", "eq", "rel"},
        default="none",
        help=LANGSTRATTYPE_HELP
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def build_metathesaurus_tables(
    umls_path: Union[str, os.PathLike],
    srdef_path: Union[str, os.PathLike],
    sg_path: Union[str, os.PathLike],
    filter_types: bool=True,
    lang: Optional[Union[str, List[str]]]=None,
    chunksize: Optional[int]=None,
    logger: Optional[logging.Logger]=None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and process data from the standard UMLS download"""
    # load SRDEF: base table for info about semantic types
    srdef_usenames = ["RT", "UI", "STY_or_RL"]
    if logger:
        logger.info("[build_metathesaurus_tables] Loading %s...", srdef_path)
    srdef = pd.read_csv(
        srdef_path, sep="|", engine="c",
        usecols=[SRDEF_COLNAMES.index(name) for name in srdef_usenames],
        names=srdef_usenames
    )
    stypes = srdef[srdef.RT == "STY"]  # here we"re only interested in STs, not relations
    if logger:
        logger.info("[build_metathesaurus_tables] Loading %s...", sg_path)
    sem_groups = pd.read_csv(  # semantic group table: we use this to map SGs to TUIs
        sg_path, sep="|", engine="c", names=["Abbrev", "Name", "TUI", "TypeName"]
    )
    tui2sg = sem_groups.set_index("TUI").Abbrev.to_dict()
    stypes["SG"] = stypes.UI.apply(lambda x: tui2sg[x])
    if filter_types:
        select_types = stypes.SG.isin(INCLUDE_GROUPS) & \
            ~stypes.STY_or_RL.isin(EXCLUDE_SEMANTIC_TYPES)
        stypes = stypes[select_types]
    mrsty_usenames = ["CUI", "TUI", "STN", "STY"]
    if logger:
        logger.info("[build_metathesaurus_tables] Loading MRSTY table for linking...")
    mrsty = pd.read_csv(  # MRSTY: links CUIs to semantic types
        os.path.join(umls_path, "MRSTY.RRF"), sep="|", engine="c",
        usecols=[MRSTY_COLNAMES.index(name) for name in mrsty_usenames],
        names=mrsty_usenames
    )
    mrsty_sg = mrsty.merge(stypes[["SG", "UI"]].rename({"UI": "TUI"}, axis=1), on="TUI")

    # add a tree depth column to the semantic type table to
    # select which type to associate with concepts that
    # belong to multiple semantic groups; prioritise more general
    # types, i.e. lower tree depth
    # for concepts with more than one lowest-depth semantic type,
    # we just use the first one to appear in the table
    if logger:
        logger.info("[build_metathesaurus_tables] Preprocessing semantic groups...")
    mrsty_sg["tree_depth"] = mrsty_sg.STN.apply(lambda x: len(x.split(".")))
    min_depths = mrsty_sg[["CUI", "tree_depth"]].groupby("CUI").tree_depth.transform(min)
    mrsty_sg = mrsty_sg[min_depths == mrsty_sg.tree_depth]
    mrsty_sg.drop_duplicates(subset=["CUI"], inplace=True)

    # load & filter all concepts, names and relations
    if logger:
        logger.info("[build_metathesaurus_tables] Loading concept table...")
    mrconso_usenames = ["CUI", "LAT", "ISPREF", "AUI", "STR"]
    chunksize = int(1e7) if chunksize is None else chunksize
    mrconso_reader = pd.read_csv(
        os.path.join(umls_path, "MRCONSO.RRF"), sep="|", engine="c",
        usecols=[MRCONSO_COLNAMES.index(name) for name in mrconso_usenames],
        names=mrconso_usenames, chunksize=chunksize
    )
    for chunk in mrconso_reader:
        if isinstance(lang, str):
            chunk = chunk[chunk.LAT == lang]
        elif isinstance(lang, list):
            chunk = chunk[chunk.LAT.isin(lang)]
        try:
            mrconso = pd.concat((mrconso, chunk))
        except NameError:
            mrconso = chunk
    mrconso = mrconso.merge(mrsty_sg.drop(["STN", "tree_depth"], axis=1), on="CUI")
    # in order to have a unique reference string for each concept,
    # we take the shortest preferred term
    # we put all the other terms in a separate table, which will not
    # necessarily have unique CUIs, to be used for SY (synonym) relations
    mrconso_prefterms = mrconso[mrconso.ISPREF == "Y"].dropna(subset=["STR"])
    mrconso_ref = mrconso_prefterms.loc[
        mrconso_prefterms.assign(str_len=mrconso_prefterms.STR.apply(len)) \
            .groupby("CUI").str_len.idxmax()
    ]
    assert mrconso_ref.CUI.drop_duplicates().count() == len(mrconso_ref)
    mrconso_other = pd.concat((
        mrconso_prefterms[~mrconso_prefterms.index.isin(mrconso_ref.index)],
        mrconso[mrconso.ISPREF != "Y"]
    )).reset_index(drop=True)
    mrconso_ref.drop(["ISPREF"], axis=1, inplace=True)

    if logger:
        logger.info("[build_metathesaurus_tables] Loading relation table...")
    mrrel_usenames = ["CUI1", "AUI1", "REL", "CUI2", "AUI2"]
    mrrel_reader = pd.read_csv(
        os.path.join(umls_path, "MRREL.RRF"), sep="|", engine="c",
        usecols=[MRREL_COLNAMES.index(name) for name in mrrel_usenames],
        names=mrrel_usenames, chunksize=chunksize
    )
    for chunk in mrrel_reader:
        chunk = chunk[chunk.CUI1.isin(mrconso_ref.CUI) & chunk.CUI2.isin(mrconso_ref.CUI)]
        try:
            mrrel = pd.concat((mrrel, chunk))
        except NameError:
            mrrel = chunk

    # add semantic groups to the relations table
    if logger:
        logger.info("[build_metathesaurus_tables] Merging with semantic groups...")
    mrconso_ref.set_index("CUI", inplace=True)
    cui2sg = mrconso_ref.SG.to_dict()
    mrrel_sg_cols = {
        "SG" + str(i + 1): [cui2sg[cui] for cui in mrrel["CUI" + str(i + 1)]] for i in range(2)
    }
    mrrel = mrrel.assign(**mrrel_sg_cols)
    return mrconso_ref, mrconso_other, mrrel.reset_index(drop=True)


def build_triple_dataset(
    mrrel: pd.DataFrame,
    mrconso_ref: pd.DataFrame,
    mrconso_other: pd.DataFrame,
    size: Optional[int]=None,
    stratify_sg: bool=True
) -> pd.DataFrame:
    """Construct the base dataset of KG triples"""
    if not size:
        size = len(mrrel)
    if stratify_sg:
        sg_counts = mrconso_ref[["SG"]].reset_index(drop=False).groupby("SG").count().iloc[:,0]
        # in case there are semantic groups that make up a small enough proportion of
        # the dataset that scaling by sample size would give zero, we add 1 to each
        # sample size and then subtract 1 from groups with more than that until we
        # have the specified total dataset size
        sg_sample_sizes = (size * (sg_counts / len(mrconso_ref)) + 1).astype(int).to_dict()
        while sum(sg_sample_sizes.values()) > size:
            for k, val in sg_sample_sizes.items():
                if sum(sg_sample_sizes.values()) == size:
                    break
                if val > 1:
                    sg_sample_sizes[k] -= 1
        for i, tuple_ in enumerate(sg_sample_sizes.items()):
            sem_grp, sample_size = tuple_
            try:
                sg_sample = mrrel[mrrel.SG2 == sem_grp].sample(sample_size)
            except ValueError:
                # sometimes the number of concepts for a given semantic group is greater than
                # the number of relations with those concepts as head entities, so `m` is too
                # big to sample without replacement from the stratified table; in this case
                # we just use all the relations available for the group in question. This will
                # not affect negative sampling, but will affect the final size of the dataset
                sg_sample = mrrel[mrrel.SG2 == sem_grp]
            if i:
                dataset = pd.concat((dataset, sg_sample))
            else:
                dataset = sg_sample
    else:
        dataset = mrrel.sample(size)
    dataset.reset_index(inplace=True, drop=True)

    # to associate strings with relations, we use the reference string from
    # the mrconso_ref index in all cases UNLESS it's a synonym relation, in
    # which case, for the target concept (CUI1), we pick another string
    # associated with the same CUI from the mrconso_other table, which has
    # all the other terms for that concept
    dataset = dataset.merge(mrconso_ref[["STR", "LAT"]], how="left", left_on="CUI2", right_on="CUI") \
        .rename({"STR": "STR2"}, axis=1)
    synonyms = dataset[dataset.REL == "SY"]
    other_rels = dataset[~dataset.index.isin(synonyms.index)]
    synonyms = synonyms.merge(
        mrconso_other[["CUI", "STR"]].drop_duplicates(subset=["CUI"]),
        how="left", left_on="CUI1", right_on="CUI"
    ).rename({"STR": "STR1"}, axis=1).drop(["CUI"], axis=1)
    # if there are no strings available other than the reference one, we reuse that
    synonyms.STR1.fillna(synonyms.STR2, inplace=True)
    other_rels = other_rels.merge(
        mrconso_ref[["STR"]], how="left", left_on="CUI1", right_on="CUI"
    ).rename({"STR": "STR1"}, axis=1)
    dataset = pd.concat((synonyms, other_rels))
    return dataset


def build_triple_classification_dataset(
    triple_dataset: pd.DataFrame,
    mrconso_ref: pd.DataFrame,
    mrrel: pd.DataFrame,
    size: Optional[int]=None
) -> pd.DataFrame:
    """Sample from the base dataset to create a dataset of triples with binary true/false labels"""
    triple_sample = triple_dataset
    if size:
        if size < len(triple_dataset):
            triple_sample = triple_dataset.sample(size)
    triple_sample["clf_label"] = [1 for _ in range(len(triple_sample))]
    sg_sample_sizes = triple_sample[["SG2"]].reset_index(drop=False).groupby("SG2") \
        .count().iloc[:, 0].to_dict()
    between_group_relations = triple_sample[triple_sample.SG1 != triple_sample.SG2]  # to be used in negative sampling strategy 2
    triple_sample.drop(["SG1", "SG2"], inplace=True, axis=1)

    # negative sampling strategy 1: sample a concept at random then sample & resample
    # from the other concepts in the same semantic group until we find one for which no
    # relation exists, then randomly select a relation to go between them
    # this makes up (roughly) the first half of the negative sampling part, i.e. 25% of the dataset
    relation_cui_check_idx = mrrel.CUI1 + mrrel.CUI2
    relations = triple_sample.REL.drop_duplicates()
    for sem_grp, sample_size in sg_sample_sizes.items():
        if sample_size == 1:
            continue  # only do this for semantic groups with more than one representative
        sg_concepts = mrconso_ref[mrconso_ref.SG == sem_grp]
        concept_sample = sg_concepts.sample(int(sample_size / 2))
        for _, row in concept_sample.iterrows():
            sample_target_concept = sg_concepts.sample().reset_index()
            while row.name + sample_target_concept.CUI.item() in relation_cui_check_idx:
                # check if a relation exists
                sample_target_concept = sg_concepts.sample().reset_index()
            neg_relation = relations.sample().item()
            # columns: CUI1, AUI1, REL, CUI2, AUI2, STR2, LAT, STR1, clf_label
            triple_sample.loc[triple_sample.index.max() + 1] = [
                sample_target_concept.CUI.item(),
                sample_target_concept.AUI.item(),
                neg_relation, row.name, row.AUI,
                row.STR, None, sample_target_concept.STR.item(), 0
            ]

    # negative sampling strategy 2: find any relation for which the two concepts come from
    # different semantic groups, then choose another two concepts from the same two semantic
    # groups for which the same relation does not exist, and use that as a negatively-labelled 
    # triple
    relation_full_check_idx = relation_cui_check_idx + mrrel.REL
    while len(triple_sample) < size * 2:
        relation_sample = between_group_relations.sample()
        rel = relation_sample.REL.item()
        sample_concept1 = mrconso_ref[mrconso_ref.SG == relation_sample.SG1.item()] \
            .sample()
        cui1 = sample_concept1.index.item()
        sample_concept2 = mrconso_ref[mrconso_ref.SG == relation_sample.SG2.item()] \
            .sample()
        while cui1 + sample_concept2.index.item() + rel in relation_full_check_idx:
            sample_concept2 = mrconso_ref[mrconso_ref.SG == relation_sample.SG2.item()] \
                .sample()
        triple_sample.loc[triple_sample.index.max() + 1] = [
            cui1, sample_concept1.AUI.item(), rel,
            sample_concept2.index.item(),
            sample_concept2.AUI.item(),
            sample_concept1.STR.item(), None,
            sample_concept2.STR.item(), 0
        ]
    return triple_sample.sample(frac=1.)


def build_path_dataset(
    triple_dataset: pd.DataFrame,
    size: int,
    max_path_len: int,
    stdout_updates: bool=False
) -> Dict[str, Dict[str, str]]:
    """Constructs a dataset of `semantic paths` for the link prediction subtask"""
    triple_dataset_pathselect = triple_dataset[triple_dataset.REL != "SY"].sample(frac=1.).reset_index(drop=True)
    total_rows = len(triple_dataset_pathselect)
    path_dataset = {}
    path_id = 0
    if size > len(triple_dataset):
        size = len(triple_dataset)
    for i, sample in triple_dataset_pathselect.iterrows():
        triple = sample.to_dict()
        cui_path = [triple["CUI1"]]
        path = {"t0": {k: triple[k] for k in ("STR2", "REL", "STR1")}}
        while len(path) < max_path_len:
            prev_cui = triple["CUI1"]
            cui_path.append(prev_cui)
            possible_next_steps = triple_dataset_pathselect[
                triple_dataset_pathselect.CUI2 == prev_cui
            ]
            next_step_iter, n_possible = 0, len(possible_next_steps)
            if not n_possible:
                break  # silently stops here if no possible next step is found
            while triple["CUI1"] in cui_path:
                triple = possible_next_steps.sample().reset_index().transpose().iloc[:, 0].to_dict()
                next_step_iter += 1
                if next_step_iter > n_possible:
                    break
            else:
                path["t" + str(len(path))] = {"REL": triple["REL"], "STR": triple["STR1"]}
        if len(path) > 1:
            path_dataset[path_id] = path
            path_id += 1
            if stdout_updates:
                sys.stdout.write("\r")
                sys.stdout.flush()
                sys.stdout.write(f"N. paths: {len(path_dataset)} / {size} ({i} / {total_rows} rows tried)")
                sys.stdout.flush()
        if len(path_dataset) == size:
            break
    return path_dataset


def main(args: Namespace, logger: logging.Logger) -> None:
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
    monolingual = len(args.lang) == 1
    if monolingual:
        args.lang = args.lang.pop()
        subdir_stem = args.lang.lower() + "_v"
    else:
        subdir_stem = "multi_v"
    datagen_version = 0
    while subdir_stem + str(datagen_version) in os.listdir(args.writepath):
        datagen_version += 1
    subdir = subdir_stem + str(datagen_version)
    write_dir = os.path.join(args.writepath, subdir)
    os.mkdir(write_dir)
    if args.load_base_tables is None:
        mrconso_ref, mrconso_other, mrrel = build_metathesaurus_tables(
            args.umls_dir, args.srdef, args.sg, lang=args.lang,
            logger=None if args.verbose else logger
        )
        bt_dir = os.path.join(write_dir, "base_metathesaurus_tables")
        logger.info("Base tables constructed, writing out to %s...", bt_dir)
        os.mkdir(bt_dir)
        mrconso_ref.to_csv(os.path.join(bt_dir, "concept_ref.tsv"), sep="\t")
        mrconso_other.to_csv(os.path.join(bt_dir, "concepts.tsv"), sep="\t", index=False)
        mrrel.to_csv(os.path.join(bt_dir, "relations.tsv"), sep="\t", index=False)
    else:
        logger.info("Loading metathesaurus tables from %s", args.load_base_tables)
        kwargs = {"sep": "\t", "engine": "pyarrow", "dtype_backend": "pyarrow"}
        mrconso_ref = pd.read_csv(
            os.path.join(args.load_base_tables, "concept_ref.tsv"),
            index_col="CUI", **kwargs
        )
        mrconso_other = pd.read_csv(
            os.path.join(args.load_base_tables, "concepts.tsv"),
            **kwargs
        )
        mrrel = pd.read_csv(os.path.join(args.load_base_tables, "relations.tsv"), **kwargs)

    logger.info(f"Writing to %s...", write_dir)
    # entity prediction: task index 0
    triple_dataset = build_triple_dataset(
        mrrel, mrconso_ref, mrconso_other, size=args.n_samples_base
    )
    langstrat = not (monolingual or args.lang_strat_type == "none")
    if langstrat:
        if args.lang_strat_type == "rel":
            lang_sample_sizes = [
                int(args.n_samples * (triple_dataset.LAT == l).sum() / len(triple_dataset)) \
                    for l in args.lang
            ]
        else:
            lang_sample_sizes = [int(args.n_samples / len(args.lang))] * len(args.lang)
        if len(triple_dataset) % len(args.lang) > 0:
            lang_sample_sizes[-1] += 1
        triple_dataset_sample_list = []
        for lang, sample_size in zip(args.lang, lang_sample_sizes):
            lang_triples = triple_dataset[triple_dataset.LAT == lang]
            if len(lang_triples) > sample_size:
                lang_triples = lang_triples.sample(sample_size)
            triple_dataset_sample_list.append(lang_triples)
        triple_dataset_sampled = pd.concat(triple_dataset_sample_list)
    else:
        triple_dataset_sampled = triple_dataset.sample(args.n_samples)
    logger.info("Triple dataset for entity prediction: n=%d", len(triple_dataset_sampled))
    triple_dataset_sampled.to_csv(os.path.join(write_dir, "triples.tsv"), sep="\t", index=False)
    
    # triple classification: task index 2
    logger.info("Building triple dataset for classification...")
    triple_classification_dataset = build_triple_classification_dataset(
        triple_dataset_sampled if langstrat else triple_dataset,
        mrconso_ref, mrrel, size=args.n_samples
    )
    logger.info("Triple dataset for classification: n=%d", len(triple_classification_dataset))
    triple_classification_dataset.to_csv(
        os.path.join(write_dir, "triple_clf.tsv"), sep="\t", index=False
    )
    
    # link prediction: task index 1
    logger.info("Building path dataset for link prediction...")
    path_dataset = build_path_dataset(
        triple_dataset[triple_dataset.LAT.isin(args.lang)] if langstrat else triple_dataset,
        args.n_samples,
        args.max_path_len,
        stdout_updates=args.verbose
    )
    logger.info("Path dataset for link prediction: n=%d", len(path_dataset))
    with open(
        os.path.join(write_dir, "paths.json"), "w", encoding=sys.getdefaultencoding()
    ) as f_io:
        json.dump(path_dataset, f_io)


if __name__ == "__main__":
    args_ = parse_arguments()
    logger_ = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - \t%(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO if args_.verbose else logging.WARNING
    )
    main(args_, logger_)
