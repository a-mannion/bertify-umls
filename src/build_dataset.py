import os
import sys
import json
import warnings
from argparse import ArgumentParser
from random import choice

import pandas as pd


UMLS_DIR_HELP = """Path to the directory containing the base UMLS dataset (should end with
`2022AB/META`)"""
SRDEF_PATH_HELP = """Path to the basic information file for semantic types and relations (SRDEF
from the UMLS semantic network )""" 
SG_PATH_HELP = """Path to the semantic groups .txt file"""
WRITE_PATH_HELP = """Directory in which to write the output datasets"""

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
EXCLUDE_SEMANTIC_TYPES = "Plant", "Fungus", "Animal", "Vertebrate", \
    "Amphibian", "Bird", "Fish", "Reptile", "Mammal", "Event", \
    "Activity", "Social Behaviour", "Daily or Recreational Activity", \
    "Occupational Activity", "Governmental or Regulatory Activity", \
    "Educational Activity", "Machine Activity", "Conceptual Entity", \
    "Geographic Area", "Regulation or Law", "Organization", \
    "Professional Society", "Self-help or Relief Organization", \
    "Professional or Occupational Group", "Family Group", \
    "Chemical Viewed Structurally", "Intellectual Product", \
    "Language", "Eukaryote"


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("umls_dir", type=str, help=UMLS_DIR_HELP)
    parser.add_argument("srdef", type=str, help=SRDEF_PATH_HELP)
    parser.add_argument("sg", type=str, help=SG_PATH_HELP)
    parser.add_argument("writepath", type=str, default=WRITE_PATH_HELP)
    parser.add_argument("--lang", type=str, default="FRE")
    parser.add_argument("--load_base_tables", type=str)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_samples_base", type=int)
    parser.add_argument("--max_path_len", type=int, default=10)
    parser.add_argument("--save_base_tables", action="store_true")
    parser.add_argument("--paths_only", action="store_true")
    return parser.parse_args()


def build_metathesaurus_tables(umls_path, srdef_path, sg_path, filter_types=True, lang=None, chunksize=None):
    # load SRDEF: base table for info about semantic types
    srdef_usenames = ["RT", "UI", "STY_or_RL"]
    srdef = pd.read_csv(
        srdef_path, sep="|",
        usecols=[SRDEF_COLNAMES.index(name) for name in srdef_usenames],
        names=srdef_usenames
    )
    stypes = srdef[srdef.RT == "STY"]  # here we"re only interested in STs, not relations
    sem_groups = pd.read_csv(  # semantic group table: we use this to map SGs to TUIs
        sg_path, sep="|", names=["Abbrev", "Name", "TUI", "TypeName"]
    )
    tui2sg = sem_groups.set_index("TUI").Abbrev.to_dict()
    stypes["SG"] = stypes.UI.apply(lambda x: tui2sg[x])
    if filter_types:
        select_types = stypes.SG.isin(INCLUDE_GROUPS) & \
            ~stypes.STY_or_RL.isin(EXCLUDE_SEMANTIC_TYPES)
        stypes = stypes[select_types]
    mrsty_usenames = ["CUI", "TUI", "STN", "STY"]
    mrsty = pd.read_csv(  # MRSTY: links CUIs to semantic types
        os.path.join(umls_path, "MRSTY.RRF"), sep="|",
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
    mrsty_sg["tree_depth"] = mrsty_sg.STN.apply(lambda x: len(x.split(".")))
    min_depths = mrsty_sg[["CUI", "tree_depth"]].groupby("CUI").tree_depth.transform(min)
    mrsty_sg = mrsty_sg[min_depths == mrsty_sg.tree_depth]
    mrsty_sg.drop_duplicates(subset=["CUI"], inplace=True)

    # load & filter all concepts, names and relations
    mrconso_usenames = ["CUI", "LAT", "ISPREF", "AUI", "STR"]
    chunksize = int(1e7) if chunksize is None else chunksize
    mrconso_reader = pd.read_csv(
        os.path.join(umls_path, "MRCONSO.RRF"), sep="|",
        usecols=[MRCONSO_COLNAMES.index(name) for name in mrconso_usenames],
        names=mrconso_usenames, chunksize=chunksize
    )
    for chunk in mrconso_reader:
        if lang:
            chunk = chunk[chunk.LAT == lang]
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

    mrrel_usenames = ["CUI1", "AUI1", "REL", "CUI2", "AUI2"]
    mrrel_reader = pd.read_csv(
        os.path.join(umls_path, "MRREL.RRF"), sep="|",
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
    mrconso_ref.set_index("CUI", inplace=True)
    cui2sg = mrconso_ref.SG.to_dict()
    mrrel_sg_cols = {
        "SG" + str(i + 1): [cui2sg[cui] for cui in mrrel["CUI" + str(i + 1)]] for i in range(2)
    }
    mrrel = mrrel.assign(**mrrel_sg_cols)
    return mrconso_ref, mrconso_other, mrrel.reset_index(drop=True)


def build_triple_dataset(mrrel, mrconso_ref, mrconso_other, size=None, stratify_sg=True):
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
    dataset = dataset.merge(mrconso_ref[["STR"]], how="left", left_on="CUI2", right_on="CUI") \
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


def build_triple_classification_dataset(triple_dataset, mrconso_ref, mrrel, size=None):
    triple_sample = triple_dataset.sample(size) if size else triple_dataset
    triple_sample["clf_label"] = [1 for _ in range(len(triple_sample))]
    sg_sample_sizes = triple_sample[["SG2"]].reset_index(drop=False).groupby("SG2") \
        .count().iloc[:,0].to_dict()
    # to be used in negative sampling strategy 2
    between_group_relations = triple_sample[triple_sample.SG1 != triple_sample.SG2]
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
            sample_target_concept = sg_concepts.sample().reset_index().transpose() \
                .iloc[:, 0].to_dict()
            while row.name + sample_target_concept["CUI"] in relation_cui_check_idx:
                # check if a relation exists
                sample_target_concept = sg_concepts.sample().reset_index().transpose() \
                    .iloc[:, 0].to_dict()
            neg_relation = relations.sample().item()
            triple_sample.loc[triple_sample.index.max() + 1] = [
                sample_target_concept["CUI"],
                sample_target_concept["AUI"],
                neg_relation, row.name, row.AUI,
                row.STR, sample_target_concept["STR"], 0
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
            sample_concept1.STR.item(),
            sample_concept2.STR.item(), 0
        ]
    return triple_sample.sample(frac=1.)


def build_path_dataset(triple_dataset, size, max_path_len):
    triple_dataset_pathselect = triple_dataset[triple_dataset.REL != "SY"]
    path_dataset = {}
    path_id = 0
    for _, sample in triple_dataset_pathselect.iterrows():
        triple = sample.to_dict()
        cui_path = [triple["CUI1"]]
        path_len = choice(range(3, max_path_len)) if max_path_len > 3 else max_path_len
        path = {"t0": {k: triple[k] for k in ("STR2", "REL", "STR1")}}
        while len(path) < path_len:
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
        if len(path_dataset) == size:
            break
    return path_dataset


def main(args):
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
    subdir_stem = args.lang.lower() + "_v"
    datagen_version = 0
    while subdir_stem + str(datagen_version) in os.listdir(args.writepath):
        datagen_version += 1
    subdir = subdir_stem + str(datagen_version)
    write_dir = os.path.join(args.writepath, subdir)
    os.mkdir(write_dir)
    if args.load_base_tables is None:
        mrconso_ref, mrconso_other, mrrel = build_metathesaurus_tables(
            args.umls_dir, args.srdef, args.sg, lang=args.lang
        )
        if args.save_base_tables:
            bt_dir = os.path.join(write_dir, "base_metathesaurus_tables")
            os.mkdir(bt_dir)
            mrconso_ref.to_csv(os.path.join(bt_dir, "concept_ref.tsv"), sep="\t")
            mrconso_other.to_csv(os.path.join(bt_dir, "concepts.tsv"), sep="\t", index=False)
            mrrel.to_csv(os.path.join(bt_dir, "relations.tsv"), sep="\t", index=False)
    else:
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

    print(f"Writing to {write_dir}...")
    triple_dataset = build_triple_dataset(
        mrrel, mrconso_ref, mrconso_other, size=args.n_samples_base
    )
    if not args.paths_only:
        print(f"Triple dataset for entity prediction: n={args.n_samples}")
        triple_dataset.sample(args.n_samples) \
            .to_csv(os.path.join(write_dir, "triples.tsv"), sep="\t", index=False)
        triple_dataset.to_csv(os.path.join(write_dir, "triples.tsv"), sep="\t", index=False)
        triple_classification_dataset = build_triple_classification_dataset(
            triple_dataset, mrconso_ref, mrrel, size=args.n_samples
        )
        print(f"Triple dataset for classification: n={len(triple_classification_dataset)}")
        triple_classification_dataset.to_csv(
            os.path.join(write_dir, "triple_clf.tsv"), sep="\t", index=False
        )

    path_dataset = build_path_dataset(triple_dataset, args.n_samples, args.max_path_len)
    print(f"Path dataset for link prediction: n={len(path_dataset)}")
    with open(
        os.path.join(write_dir, "paths.json"), "w", encoding=sys.getdefaultencoding()
    ) as f_io:
        json.dump(path_dataset, f_io)


if __name__ == "__main__":
    main(parse_arguments())
