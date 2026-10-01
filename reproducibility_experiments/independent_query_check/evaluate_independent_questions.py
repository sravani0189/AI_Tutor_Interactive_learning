import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi


# ============================================================
# FILES
# ============================================================

DATASET_FILE = "Merged_Chapter_Dataset.csv"
QUESTION_FILE = "independent_learner_questions_filled.csv"

RESULT_FILE = "independent_learner_retrieval_results.csv"
SUMMARY_FILE = "independent_learner_retrieval_summary.json"


# ============================================================
# FROZEN RETRIEVAL SETTINGS
# These are NOT tuned on the 43 independent questions.
# ============================================================

BM25_K1 = 0.8
BM25_B = 0.9
HYBRID_ALPHA = 0.8

K_VALUES = [1, 3, 5]


print()
print("==============================")
print("INDEPENDENT LEARNER RETRIEVAL")
print("==============================")


# ============================================================
# LOAD FILES
# ============================================================

for filename in [DATASET_FILE, QUESTION_FILE]:
    if not Path(filename).exists():
        raise FileNotFoundError(
            f"Missing required file: {filename}"
        )


dataset = pd.read_csv(DATASET_FILE)
questions = pd.read_csv(QUESTION_FILE)


print("Dataset rows:", len(dataset))
print("Independent questions:", len(questions))


# ============================================================
# VALIDATE COLUMNS
# ============================================================

dataset_required = [
    "chapter",
    "Chapter Content",
]

question_required = [
    "question_id",
    "top_chapter",
    "source_section",
    "independent_learner_question",
]


for col in dataset_required:
    if col not in dataset.columns:
        raise RuntimeError(
            f"Dataset missing column: {col}"
        )


for col in question_required:
    if col not in questions.columns:
        raise RuntimeError(
            f"Question file missing column: {col}"
        )


# ============================================================
# CLEAN DATA
# ============================================================

dataset["chapter"] = (
    dataset["chapter"]
    .astype(str)
    .str.strip()
)

dataset["Chapter Content"] = (
    dataset["Chapter Content"]
    .fillna("")
    .astype(str)
)


questions["source_section"] = (
    questions["source_section"]
    .astype(str)
    .str.strip()
)

questions["independent_learner_question"] = (
    questions["independent_learner_question"]
    .fillna("")
    .astype(str)
    .str.strip()
)


if len(questions) != 43:
    raise RuntimeError(
        f"Expected 43 independent questions, found {len(questions)}."
    )


if (
    questions["independent_learner_question"] == ""
).any():
    raise RuntimeError(
        "Some independent learner questions are blank."
    )


# ============================================================
# BUILD SECTION-LEVEL CORPUS
# ============================================================

corpus = (
    dataset[
        ["chapter", "Chapter Content"]
    ]
    .drop_duplicates(subset=["chapter"])
    .reset_index(drop=True)
)


print("Retrieval sections:", len(corpus))


# Section 11.1 had missing source content in the archived dataset.
# It is not part of the 43 independent questions.
empty_sections = corpus[
    corpus["Chapter Content"].str.strip() == ""
]["chapter"].tolist()

if empty_sections:
    print(
        "Sections with blank source text:",
        empty_sections
    )


# ============================================================
# VERIFY GOLD SECTIONS
# ============================================================

available_sections = set(
    corpus["chapter"]
)

missing_gold = sorted(
    set(questions["source_section"])
    - available_sections
)

if missing_gold:
    raise RuntimeError(
        "Gold sections missing from corpus: "
        + str(missing_gold)
    )


# ============================================================
# TF-IDF
# Frozen specification:
# word unigrams + bigrams
# sublinear TF
# L2 normalization
# ============================================================

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    norm="l2"
)

tfidf_matrix = tfidf.fit_transform(
    corpus["Chapter Content"]
)


# ============================================================
# BM25
# ============================================================

def tokenize(text):
    return str(text).lower().split()


tokenized_corpus = [
    tokenize(text)
    for text in corpus["Chapter Content"]
]


bm25 = BM25Okapi(
    tokenized_corpus,
    k1=BM25_K1,
    b=BM25_B
)


# ============================================================
# HELPERS
# ============================================================

def minmax(values):

    values = np.asarray(
        values,
        dtype=float
    )

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return np.zeros_like(values)

    return (
        (values - minimum)
        / (maximum - minimum)
    )


def get_rank(ranked_indices, gold_index):

    positions = np.where(
        ranked_indices == gold_index
    )[0]

    if len(positions) == 0:
        return None

    return int(
        positions[0] + 1
    )


# ============================================================
# RUN RETRIEVAL
# ============================================================

results = []


for _, row in questions.iterrows():

    qid = row["question_id"]
    query = row["independent_learner_question"]
    gold_section = row["source_section"]


    gold_matches = np.where(
        corpus["chapter"].to_numpy()
        == gold_section
    )[0]


    if len(gold_matches) != 1:
        raise RuntimeError(
            f"Could not uniquely identify gold section "
            f"for {qid}: {gold_section}"
        )


    gold_index = int(
        gold_matches[0]
    )


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    query_vector = tfidf.transform(
        [query]
    )

    tfidf_scores = (
        tfidf_matrix
        @ query_vector.T
    ).toarray().ravel()

    tfidf_ranking = np.argsort(
        -tfidf_scores
    )


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_scores = np.asarray(
        bm25.get_scores(
            tokenize(query)
        ),
        dtype=float
    )

    bm25_ranking = np.argsort(
        -bm25_scores
    )


    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    tfidf_normalized = minmax(
        tfidf_scores
    )

    bm25_normalized = minmax(
        bm25_scores
    )

    hybrid_scores = (
        HYBRID_ALPHA * tfidf_normalized
        + (1.0 - HYBRID_ALPHA)
        * bm25_normalized
    )

    hybrid_ranking = np.argsort(
        -hybrid_scores
    )


    # --------------------------------------------------------
    # GOLD RANKS
    # --------------------------------------------------------

    tfidf_rank = get_rank(
        tfidf_ranking,
        gold_index
    )

    bm25_rank = get_rank(
        bm25_ranking,
        gold_index
    )

    hybrid_rank = get_rank(
        hybrid_ranking,
        gold_index
    )


    result = {
        "question_id": qid,
        "top_chapter": int(row["top_chapter"]),
        "gold_section": gold_section,
        "question": query,

        "tfidf_rank": tfidf_rank,
        "bm25_rank": bm25_rank,
        "hybrid_rank": hybrid_rank,

        "tfidf_top1_section":
            corpus.iloc[
                tfidf_ranking[0]
            ]["chapter"],

        "bm25_top1_section":
            corpus.iloc[
                bm25_ranking[0]
            ]["chapter"],

        "hybrid_top1_section":
            corpus.iloc[
                hybrid_ranking[0]
            ]["chapter"],
    }


    for method, ranking in [
        ("tfidf", tfidf_ranking),
        ("bm25", bm25_ranking),
        ("hybrid", hybrid_ranking),
    ]:

        for k in K_VALUES:

            result[
                f"{method}_recall_at_{k}"
            ] = int(
                gold_index
                in ranking[:k]
            )


    results.append(result)


results = pd.DataFrame(results)


# ============================================================
# METRICS
# ============================================================

def recall(method, k):

    return float(
        results[
            f"{method}_recall_at_{k}"
        ].mean()
    )


def mrr(rank_column):

    reciprocal_ranks = []

    for rank in results[rank_column]:

        if pd.isna(rank):
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(
                1.0 / float(rank)
            )

    return float(
        np.mean(reciprocal_ranks)
    )


summary = {

    "benchmark":
        "independently_worded_learner_style_questions",

    "n_questions":
        int(len(results)),

    "n_retrieval_sections":
        int(len(corpus)),

    "question_construction":
        (
            "Questions constructed from isolated textbook "
            "source text without access to prepared QA pairs."
        ),

    "frozen_parameters": {
        "tfidf_ngram_range": [1, 2],
        "tfidf_sublinear_tf": True,
        "tfidf_norm": "l2",
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "hybrid_alpha": HYBRID_ALPHA,
    },

    "results": {},

    "chapter_distribution": {
        str(chapter): int(count)
        for chapter, count
        in questions[
            "top_chapter"
        ]
        .value_counts()
        .sort_index()
        .items()
    }
}


for method in [
    "tfidf",
    "bm25",
    "hybrid"
]:

    summary["results"][method] = {

        "Recall@1":
            recall(method, 1),

        "Recall@3":
            recall(method, 3),

        "Recall@5":
            recall(method, 5),

        "MRR":
            mrr(
                f"{method}_rank"
            ),
    }


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("==============================")
print("INDEPENDENT QUERY RESULTS")
print("==============================")


for method in [
    "tfidf",
    "bm25",
    "hybrid"
]:

    print()
    print(method.upper())

    print(
        "Recall@1:",
        f"{recall(method, 1):.4f}"
    )

    print(
        "Recall@3:",
        f"{recall(method, 3):.4f}"
    )

    print(
        "Recall@5:",
        f"{recall(method, 5):.4f}"
    )

    print(
        "MRR:",
        f"{mrr(f'{method}_rank'):.4f}"
    )


print()
print("==============================")
print("CHAPTER DISTRIBUTION")
print("==============================")


for chapter, count in (
    questions[
        "top_chapter"
    ]
    .value_counts()
    .sort_index()
    .items()
):

    print(
        f"Chapter {int(chapter)}: "
        f"{int(count)}"
    )


print()
print("==============================")
print("FILES SAVED")
print("==============================")

print(RESULT_FILE)
print(SUMMARY_FILE)

print()
print(
    "Independent learner-query "
    "retrieval evaluation complete."
)