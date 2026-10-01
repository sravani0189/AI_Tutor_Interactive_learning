import ast
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi


# ============================================================
# SETTINGS — SAME AS OUR FINAL RETRIEVAL BENCHMARK
# ============================================================

SEED = 42
N_BOOT = 5000

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR_TF = True
TFIDF_NORM = "l2"

BM25_K1 = 0.8
BM25_B = 0.9

HYBRID_ALPHA = 0.8

TOP_K = [1, 3, 5]

CSV_FILE = "Merged_Chapter_Dataset_recovered.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

print("DATASET")
print("Rows:", len(df))
print("Sections:", df["chapter"].nunique())


def parse_list(x):
    if pd.isna(x):
        return []
    return ast.literal_eval(x)


# ============================================================
# BUILD DOCUMENT CORPUS
# ============================================================

documents = []
sections = []
section_to_index = {}

for _, row in df.iterrows():

    section = str(row["chapter"]).strip()
    content = str(row["Chapter Content"]).strip()

    if not content:
        raise ValueError(
            f"Missing source text for section: {section}"
        )

    section_to_index[section] = len(documents)

    sections.append(section)
    documents.append(content)


# ============================================================
# BUILD QUERY SET
# ============================================================

queries = []

for _, row in df.iterrows():

    section = str(row["chapter"]).strip()

    questions = parse_list(row["Questions"])
    answers = parse_list(row["Answers"])

    if len(questions) != len(answers):
        raise ValueError(
            f"Question/answer mismatch in {section}"
        )

    gold = section_to_index[section]

    for question in questions:

        queries.append(
            {
                "query": str(question).strip(),
                "section": section,
                "gold": gold,
            }
        )


print("Documents:", len(documents))
print("Usable queries:", len(queries))


# ============================================================
# SAME 80/20 SPLIT AS FINAL BENCHMARK
# ============================================================

indices = np.arange(len(queries))

dev_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
)

dev_queries = [queries[i] for i in dev_indices]
test_queries = [queries[i] for i in test_indices]

print("Development queries:", len(dev_queries))
print("Held-out test queries:", len(test_queries))


# ============================================================
# SECTION DISTRIBUTION
# ============================================================

dev_counts = {}

for item in dev_queries:
    section = item["section"]
    dev_counts[section] = dev_counts.get(section, 0) + 1


test_counts = {}

for item in test_queries:
    section = item["section"]
    test_counts[section] = test_counts.get(section, 0) + 1


distribution_rows = []

for section in sections:

    total = dev_counts.get(section, 0) + test_counts.get(section, 0)

    distribution_rows.append(
        {
            "section": section,
            "total_queries": total,
            "development_queries": dev_counts.get(section, 0),
            "test_queries": test_counts.get(section, 0),
        }
    )


distribution = pd.DataFrame(distribution_rows)

distribution.to_csv(
    "retrieval_query_section_distribution.csv",
    index=False,
)

print("\nSECTION DISTRIBUTION")
print(
    "Sections represented in development:",
    sum(distribution["development_queries"] > 0),
)
print(
    "Sections represented in test:",
    sum(distribution["test_queries"] > 0),
)
print(
    "Test sections with zero queries:",
    sum(distribution["test_queries"] == 0),
)

print("\nTest-set distribution:")
print(
    distribution[
        distribution["test_queries"] > 0
    ][
        [
            "section",
            "total_queries",
            "development_queries",
            "test_queries",
        ]
    ].to_string(index=False)
)


# ============================================================
# TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    ngram_range=TFIDF_NGRAM_RANGE,
    sublinear_tf=TFIDF_SUBLINEAR_TF,
    norm=TFIDF_NORM,
)

document_matrix = tfidf.fit_transform(documents)


def tfidf_scores(query):

    query_vector = tfidf.transform([query])

    return (
        document_matrix @ query_vector.T
    ).toarray().ravel()


# ============================================================
# BM25
# ============================================================

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(
    tokenized_documents,
    k1=BM25_K1,
    b=BM25_B,
)


def bm25_scores(query):

    return np.asarray(
        bm25.get_scores(
            query.lower().split()
        ),
        dtype=float,
    )


# ============================================================
# HYBRID
# ============================================================

def normalize(scores):

    scores = np.asarray(scores, dtype=float)

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(scores)

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


def hybrid_scores(query):

    tfidf_s = normalize(
        tfidf_scores(query)
    )

    bm25_s = normalize(
        bm25_scores(query)
    )

    return (
        HYBRID_ALPHA * tfidf_s
        + (1 - HYBRID_ALPHA) * bm25_s
    )


retrievers = {
    "TF-IDF": tfidf_scores,
    "BM25": bm25_scores,
    "Hybrid": hybrid_scores,
}


# ============================================================
# QUERY-LEVEL EVALUATION
# ============================================================

def query_metrics(query_set, scorer):

    rows = []

    for item in query_set:

        scores = scorer(item["query"])
        ranking = np.argsort(-scores)

        gold = item["gold"]

        row = {
            "query": item["query"],
            "section": item["section"],
        }

        for k in TOP_K:

            retrieved = ranking[:k]

            hit = int(gold in retrieved)

            row[f"Recall@{k}"] = hit
            row[f"Precision@{k}"] = hit / k

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# BOOTSTRAP 95% CI
# ============================================================

def bootstrap_ci(values, n_boot=N_BOOT, seed=42):

    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)

    n = len(values)

    bootstrap_means = np.empty(n_boot)

    for i in range(n_boot):

        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        bootstrap_means[i] = np.mean(sample)

    low, high = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    return float(low), float(high)


# ============================================================
# FINAL TEST RESULTS + UNCERTAINTY
# ============================================================

print("\n==============================")
print("HELD-OUT TEST RESULTS + 95% CI")
print("==============================")


summary_rows = []


for name, scorer in retrievers.items():

    test_df = query_metrics(
        test_queries,
        scorer,
    )

    test_df.to_csv(
        f"retrieval_query_results_{name.replace('-', '').replace(' ', '_')}.csv",
        index=False,
    )

    print("\n", name)

    row = {
        "method": name,
        "n_queries": len(test_df),
    }

    for k in TOP_K:

        recall_values = test_df[
            f"Recall@{k}"
        ].to_numpy()

        precision_values = test_df[
            f"Precision@{k}"
        ].to_numpy()

        recall_mean = np.mean(
            recall_values
        )

        precision_mean = np.mean(
            precision_values
        )

        recall_low, recall_high = (
            bootstrap_ci(
                recall_values,
                seed=SEED + k,
            )
        )

        precision_low, precision_high = (
            bootstrap_ci(
                precision_values,
                seed=SEED + 100 + k,
            )
        )

        row[f"Recall@{k}"] = float(
            recall_mean
        )

        row[f"Recall@{k}_CI_low"] = recall_low
        row[f"Recall@{k}_CI_high"] = recall_high

        row[f"Precision@{k}"] = float(
            precision_mean
        )

        row[f"Precision@{k}_CI_low"] = (
            precision_low
        )

        row[f"Precision@{k}_CI_high"] = (
            precision_high
        )

        print(
            f"Recall@{k}: "
            f"{recall_mean:.4f} "
            f"[{recall_low:.4f}, {recall_high:.4f}]"
        )

        print(
            f"Precision@{k}: "
            f"{precision_mean:.4f} "
            f"[{precision_low:.4f}, {precision_high:.4f}]"
        )

    summary_rows.append(row)


# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    "retrieval_test_results_with_ci.csv",
    index=False,
)

output = {
    "dataset": {
        "documents": len(documents),
        "queries": len(queries),
        "development_queries": len(dev_queries),
        "held_out_test_queries": len(test_queries),
        "random_seed": SEED,
    },
    "retrieval": {
        "retrieval_unit":
            "one textbook section",
        "chunk_overlap": 0,
        "tfidf": {
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "norm": "l2",
        },
        "bm25": {
            "k1": BM25_K1,
            "b": BM25_B,
        },
        "hybrid": {
            "alpha": HYBRID_ALPHA,
            "formula":
                "alpha*normalized_tfidf + "
                "(1-alpha)*normalized_bm25",
        },
        "top_k": TOP_K,
    },
    "test_results": summary.to_dict(
        orient="records"
    ),
}


with open(
    "retrieval_test_results_with_ci.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


print("\n==============================")
print("DONE")
print("==============================")

print(
    "Saved: retrieval_test_results_with_ci.csv"
)

print(
    "Saved: retrieval_test_results_with_ci.json"
)

print(
    "Saved: retrieval_query_section_distribution.csv"
)