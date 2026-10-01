import ast
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
TEST_SIZE = 0.20

# TF-IDF settings
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR_TF = True
TFIDF_NORM = "l2"

# Candidate BM25 settings for DEVELOPMENT tuning
BM25_K1_VALUES = [0.8, 1.2, 1.5, 1.8, 2.0]
BM25_B_VALUES = [0.2, 0.4, 0.6, 0.75, 0.9]

# Candidate hybrid weights for DEVELOPMENT tuning
HYBRID_ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4,
                       0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

TOP_K = [1, 3, 5]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("Merged_Chapter_Dataset_recovered.csv")

print("\nDATASET")
print("Rows:", len(df))
print("Sections:", df["chapter"].nunique())


def parse_list(x):
    if pd.isna(x):
        return []
    return ast.literal_eval(x)


# ============================================================
# BUILD SECTION-LEVEL CORPUS
# ============================================================

documents = []
sections = []
section_to_index = {}

for _, row in df.iterrows():

    section = str(row["chapter"]).strip()

    if pd.isna(row["Chapter Content"]):
        print("Missing source text:", section)
        continue

    content = str(row["Chapter Content"]).strip()

    if not content:
        print("Missing source text:", section)
        continue

    section_to_index[section] = len(documents)

    sections.append(section)
    documents.append(content)

print("Documents used:", len(documents))


# ============================================================
# BUILD QUERIES
# ============================================================

queries = []

for _, row in df.iterrows():

    section = str(row["chapter"]).strip()

    questions = parse_list(row["Questions"])
    answers = parse_list(row["Answers"])

    if len(questions) != len(answers):
        raise ValueError(
            f"Question/answer mismatch in {section}: "
            f"{len(questions)} vs {len(answers)}"
        )

    if section not in section_to_index:
        print(
            "Skipping",
            len(questions),
            "queries because source is missing:",
            section,
        )
        continue

    gold_doc = section_to_index[section]

    for question in questions:

        queries.append(
            {
                "query": str(question).strip(),
                "section": section,
                "gold": gold_doc,
            }
        )

print("Usable queries:", len(queries))


# ============================================================
# DEVELOPMENT / TEST SPLIT
# ============================================================

indices = np.arange(len(queries))

dev_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=SEED,
)

dev_indices = np.sort(dev_indices)
test_indices = np.sort(test_indices)

dev_queries = [queries[i] for i in dev_indices]
test_queries = [queries[i] for i in test_indices]

print("Development queries:", len(dev_queries))
print("Held-out test queries:", len(test_queries))


# ============================================================
# TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    ngram_range=TFIDF_NGRAM_RANGE,
    sublinear_tf=TFIDF_SUBLINEAR_TF,
    norm=TFIDF_NORM,
)

document_matrix = tfidf.fit_transform(documents)


def get_tfidf_scores(query):

    query_vector = tfidf.transform([query])

    scores = (
        document_matrix @ query_vector.T
    ).toarray().ravel()

    return scores


# ============================================================
# BM25 FACTORY
# ============================================================

tokenized_documents = [
    document.lower().split()
    for document in documents
]


def make_bm25(k1, b):

    return BM25Okapi(
        tokenized_documents,
        k1=k1,
        b=b,
    )


def get_bm25_scores(query, bm25):

    tokens = query.lower().split()

    return np.asarray(
        bm25.get_scores(tokens),
        dtype=float,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(scores):

    scores = np.asarray(
        scores,
        dtype=float,
    )

    minimum = scores.min()
    maximum = scores.max()

    if maximum - minimum < 1e-12:
        return np.zeros_like(scores)

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(query_set, scorer):

    hits = {1: [], 3: [], 5: []}
    precision = {1: [], 3: [], 5: []}

    for item in query_set:

        scores = scorer(item["query"])

        ranking = np.argsort(-scores)

        gold = item["gold"]

        for k in TOP_K:

            retrieved = ranking[:k]

            correct = int(
                gold in retrieved
            )

            hits[k].append(correct)

            precision[k].append(
                correct / k
            )

    results = {}

    for k in TOP_K:

        results[f"Recall@{k}"] = float(
            np.mean(hits[k])
        )

        results[f"Precision@{k}"] = float(
            np.mean(precision[k])
        )

    return results


# ============================================================
# TUNE BM25 ON DEVELOPMENT SET
# ============================================================

print("\n==============================")
print("TUNING BM25 ON DEVELOPMENT SET")
print("==============================")

best_bm25 = None
best_bm25_score = -1.0
bm25_tuning_results = []

for k1 in BM25_K1_VALUES:

    for b in BM25_B_VALUES:

        candidate = make_bm25(k1, b)

        result = evaluate(
            dev_queries,
            lambda q, model=candidate:
                get_bm25_scores(q, model),
        )

        score = result["Recall@5"]

        bm25_tuning_results.append(
            {
                "k1": k1,
                "b": b,
                **result,
            }
        )

        if score > best_bm25_score:

            best_bm25_score = score

            best_bm25 = {
                "k1": k1,
                "b": b,
            }

print(
    "Selected BM25 k1:",
    best_bm25["k1"]
)

print(
    "Selected BM25 b:",
    best_bm25["b"]
)

print(
    "Development Recall@5:",
    round(best_bm25_score, 4)
)


# ============================================================
# BUILD FINAL BM25 WITH TUNED PARAMETERS
# ============================================================

bm25 = make_bm25(
    best_bm25["k1"],
    best_bm25["b"],
)


# ============================================================
# TUNE HYBRID ALPHA ON DEVELOPMENT SET
# ============================================================

print("\n==============================")
print("TUNING HYBRID WEIGHT")
print("==============================")

best_alpha = None
best_hybrid_score = -1.0
hybrid_tuning_results = []


def get_hybrid_scores(query, alpha):

    tfidf_scores = normalize(
        get_tfidf_scores(query)
    )

    bm25_scores = normalize(
        get_bm25_scores(query, bm25)
    )

    return (
        alpha * tfidf_scores
        + (1.0 - alpha) * bm25_scores
    )


for alpha in HYBRID_ALPHA_VALUES:

    result = evaluate(
        dev_queries,
        lambda q, a=alpha:
            get_hybrid_scores(q, a),
    )

    score = result["Recall@5"]

    hybrid_tuning_results.append(
        {
            "alpha": alpha,
            **result,
        }
    )

    if score > best_hybrid_score:

        best_hybrid_score = score
        best_alpha = alpha


print(
    "Selected hybrid alpha:",
    best_alpha
)

print(
    "Development Recall@5:",
    round(best_hybrid_score, 4)
)


# ============================================================
# FINAL RETRIEVERS
# ============================================================

def final_tfidf(q):
    return get_tfidf_scores(q)


def final_bm25(q):
    return get_bm25_scores(q, bm25)


def final_hybrid(q):
    return get_hybrid_scores(q, best_alpha)


retrievers = {
    "TF-IDF": final_tfidf,
    "BM25": final_bm25,
    "Hybrid": final_hybrid,
}


# ============================================================
# DEVELOPMENT RESULTS WITH FINAL SETTINGS
# ============================================================

print("\n==============================")
print("FINAL DEVELOPMENT RESULTS")
print("==============================")

development_results = {}

for name, scorer in retrievers.items():

    result = evaluate(
        dev_queries,
        scorer,
    )

    development_results[name] = result

    print("\n" + name)

    for metric, value in result.items():

        print(
            metric,
            ":",
            round(value, 4),
        )


# ============================================================
# HELD-OUT TEST RESULTS
# ============================================================

print("\n==============================")
print("HELD-OUT TEST RESULTS")
print("==============================")

test_results = {}

for name, scorer in retrievers.items():

    result = evaluate(
        test_queries,
        scorer,
    )

    test_results[name] = result

    print("\n" + name)

    for metric, value in result.items():

        print(
            metric,
            ":",
            round(value, 4),
        )


# ============================================================
# SAVE CONFIGURATION + RESULTS
# ============================================================

output = {

    "configuration": {

        "random_seed": SEED,

        "test_size": TEST_SIZE,

        "retrieval_unit":
            "one textbook section",

        "chunk_overlap": 0,

        "tfidf": {
            "ngram_range":
                list(TFIDF_NGRAM_RANGE),
            "sublinear_tf":
                TFIDF_SUBLINEAR_TF,
            "norm":
                TFIDF_NORM,
        },

        "bm25": {
            "selected_k1":
                best_bm25["k1"],
            "selected_b":
                best_bm25["b"],
            "candidate_k1_values":
                BM25_K1_VALUES,
            "candidate_b_values":
                BM25_B_VALUES,
        },

        "hybrid": {
            "selected_alpha":
                best_alpha,
            "candidate_alpha_values":
                HYBRID_ALPHA_VALUES,
            "formula":
                "alpha*normalized_tfidf + "
                "(1-alpha)*normalized_bm25",
        },

        "top_k":
            TOP_K,

        "documents":
            len(documents),

        "usable_queries":
            len(queries),

        "development_queries":
            len(dev_queries),

        "test_queries":
            len(test_queries),
    },

    "bm25_tuning_results":
        bm25_tuning_results,

    "hybrid_tuning_results":
        hybrid_tuning_results,

    "development_results":
        development_results,

    "test_results":
        test_results,
}


with open(
    "retrieval_benchmark_results.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        indent=2,
    )


print("\n==============================")
print("DONE")
print("==============================")

print(
    "Results saved to "
    "retrieval_benchmark_results.json"
)