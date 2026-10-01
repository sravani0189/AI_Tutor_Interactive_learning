import ast
import json
import random
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

# Total benchmark:
# 30 answerable + 30 unanswerable
#
# Development:
# 15 answerable + 15 unanswerable
#
# Held-out test:
# 15 answerable + 15 unanswerable
DEV_PER_CLASS = 15
TEST_PER_CLASS = 15

DATASET = "Merged_Chapter_Dataset_recovered.csv"
QUESTIONS_FILE = "abstention_questions.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Same retrieval depth used by the recovered Colab implementation.
TOP_K = 3

# Candidate thresholds.
# These are tuned only on the development set.
THRESHOLDS = np.round(
    np.arange(0.20, 0.81, 0.025),
    3
)

# Bootstrap repetitions for 95% CIs.
N_BOOT = 5000


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):
    """
    Normalize text for duplicate/overlap checks.
    """
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_list(value):
    """
    Parse list-like strings stored in the recovered CSV.
    """
    if pd.isna(value):
        return []

    return ast.literal_eval(value)


def bootstrap_ci(values, n_boot=N_BOOT, seed=42):
    """
    Percentile bootstrap 95% CI for a binary rate.

    IMPORTANT:
    The caller should pass values from the relevant class only.
    """
    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    n = len(values)

    bootstrap_means = np.empty(
        n_boot,
        dtype=float
    )

    for i in range(n_boot):

        sample = rng.choice(
            values,
            size=n,
            replace=True
        )

        bootstrap_means[i] = np.mean(
            sample
        )

    low, high = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    return float(low), float(high)


# ============================================================
# LOAD TEXTBOOK DATA
# ============================================================

print("\n==============================")
print("LOADING DATASET")
print("==============================")

df = pd.read_csv(
    DATASET
)

print(
    "Rows:",
    len(df)
)

print(
    "Sections:",
    df["chapter"].nunique()
)


# ============================================================
# BUILD SECTION-LEVEL RETRIEVAL CORPUS
#
# Same basic unit used in the reproducible lexical benchmark:
# one textbook section = one retrieval document.
# ============================================================

documents = []
sections = []

for _, row in df.iterrows():

    section = str(
        row["chapter"]
    ).strip()

    if pd.isna(
        row["Chapter Content"]
    ):
        continue

    content = str(
        row["Chapter Content"]
    ).strip()

    if not content:
        continue

    sections.append(
        section
    )

    documents.append(
        content
    )


print(
    "Textbook retrieval documents:",
    len(documents)
)


# ============================================================
# LOAD INDEPENDENT ABSTENTION QUESTIONS
# ============================================================

print("\n==============================")
print("LOADING ABSTENTION QUESTIONS")
print("==============================")

qdf = pd.read_csv(
    QUESTIONS_FILE
)


required_columns = {
    "id",
    "label",
    "chapter",
    "question"
}

missing_columns = (
    required_columns
    - set(qdf.columns)
)

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(
            sorted(missing_columns)
        )
    )


qdf["id"] = (
    qdf["id"]
    .astype(str)
    .str.strip()
)

qdf["label"] = (
    qdf["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

qdf["chapter"] = (
    qdf["chapter"]
    .astype(str)
    .str.strip()
)

qdf["question"] = (
    qdf["question"]
    .astype(str)
    .str.strip()
)


# ============================================================
# VALIDATE LABELS
# ============================================================

valid_labels = {
    "answerable",
    "unanswerable"
}

invalid_labels = (
    set(qdf["label"])
    - valid_labels
)

if invalid_labels:

    raise ValueError(
        "Invalid labels found: "
        + str(invalid_labels)
    )


# ============================================================
# DUPLICATE QUESTION CHECK
# ============================================================

qdf["_normalized_question"] = (
    qdf["question"]
    .map(normalize_text)
)

duplicate_rows = qdf[
    qdf["_normalized_question"]
    .duplicated(
        keep=False
    )
]

if len(duplicate_rows) > 0:

    print(
        "\nDuplicate questions found:"
    )

    print(
        duplicate_rows[
            [
                "id",
                "question"
            ]
        ].to_string(
            index=False
        )
    )

    raise ValueError(
        "Duplicate abstention questions found."
    )


# ============================================================
# CHECK EXACT OVERLAP WITH INDEXED GENERATED QA QUESTIONS
#
# The recovered dataset contains the original generated
# question set. We use it only to detect exact leakage.
# ============================================================

indexed_questions = set()

for _, row in df.iterrows():

    questions = parse_list(
        row["Questions"]
    )

    for question in questions:

        indexed_questions.add(
            normalize_text(
                question
            )
        )


qdf["matches_indexed_qa"] = (
    qdf["_normalized_question"]
    .isin(indexed_questions)
)

overlap = qdf[
    qdf["matches_indexed_qa"]
]

print(
    "Exact benchmark/QA overlap:",
    len(overlap)
)

if len(overlap) > 0:

    print(
        overlap[
            [
                "id",
                "label",
                "question"
            ]
        ].to_string(
            index=False
        )
    )

    raise ValueError(
        "Remove questions overlapping "
        "with the indexed generated QA set."
    )


# ============================================================
# CLASS COUNTS
# ============================================================

answerable = qdf[
    qdf["label"]
    == "answerable"
].copy()

unanswerable = qdf[
    qdf["label"]
    == "unanswerable"
].copy()


print(
    "Answerable questions:",
    len(answerable)
)

print(
    "Unanswerable questions:",
    len(unanswerable)
)


required_per_class = (
    DEV_PER_CLASS
    +
    TEST_PER_CLASS
)

if len(answerable) < required_per_class:

    raise RuntimeError(
        "Need at least "
        f"{required_per_class} answerable questions."
    )

if len(unanswerable) < required_per_class:

    raise RuntimeError(
        "Need at least "
        f"{required_per_class} unanswerable questions."
    )


# ============================================================
# BALANCED RANDOM SPLIT
# ============================================================

rng = random.Random(
    SEED
)

answerable_records = (
    answerable
    .drop(
        columns=[
            "_normalized_question",
            "matches_indexed_qa"
        ]
    )
    .to_dict(
        "records"
    )
)

unanswerable_records = (
    unanswerable
    .drop(
        columns=[
            "_normalized_question",
            "matches_indexed_qa"
        ]
    )
    .to_dict(
        "records"
    )
)


rng.shuffle(
    answerable_records
)

rng.shuffle(
    unanswerable_records
)


answerable_records = (
    answerable_records[
        :required_per_class
    ]
)

unanswerable_records = (
    unanswerable_records[
        :required_per_class
    ]
)


development = (
    answerable_records[
        :DEV_PER_CLASS
    ]
    +
    unanswerable_records[
        :DEV_PER_CLASS
    ]
)

test = (
    answerable_records[
        DEV_PER_CLASS:
    ]
    +
    unanswerable_records[
        DEV_PER_CLASS:
    ]
)


rng.shuffle(
    development
)

rng.shuffle(
    test
)


print("\n==============================")
print("QUESTION SPLIT")
print("==============================")

print(
    "Development:",
    len(development)
)

print(
    "Held-out test:",
    len(test)
)

print(
    "Development answerable:",
    sum(
        x["label"] == "answerable"
        for x in development
    )
)

print(
    "Development unanswerable:",
    sum(
        x["label"] == "unanswerable"
        for x in development
    )
)

print(
    "Test answerable:",
    sum(
        x["label"] == "answerable"
        for x in test
    )
)

print(
    "Test unanswerable:",
    sum(
        x["label"] == "unanswerable"
        for x in test
    )
)


# ============================================================
# LOAD MINILM
# ============================================================

print("\n==============================")
print("LOADING MINILM")
print("==============================")

print(
    "Model:",
    MODEL_NAME
)

model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# ENCODE TEXTBOOK DOCUMENTS
#
# Unit-normalized embeddings mean the dot product corresponds
# to cosine similarity.
# ============================================================

print(
    "\nEncoding textbook documents..."
)

document_embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    similarities = (
        document_embeddings
        @ query_embedding
    )

    ranking = np.argsort(
        -similarities
    )

    top_indices = (
        ranking[
            :TOP_K
        ]
    )

    top_scores = [
        float(
            similarities[i]
        )
        for i in top_indices
    ]

    return {
        "top_indices":
            top_indices,

        "top_scores":
            top_scores,

        "max_similarity":
            float(
                similarities[
                    top_indices[0]
                ]
            )
    }


# ============================================================
# SCORE QUESTIONS
# ============================================================

def score_questions(records):

    rows = []

    for record in records:

        result = retrieve(
            record["question"]
        )

        top_indices = (
            result["top_indices"]
        )

        top_scores = (
            result["top_scores"]
        )

        top1 = int(
            top_indices[0]
        )

        rows.append(
            {
                "id":
                    record["id"],

                "label":
                    record["label"],

                "chapter":
                    record["chapter"],

                "question":
                    record["question"],

                "max_similarity":
                    result[
                        "max_similarity"
                    ],

                "top1_section":
                    sections[top1],

                "top1_score":
                    top_scores[0],

                "top2_score":
                    top_scores[1],

                "top3_score":
                    top_scores[2]
            }
        )

    return pd.DataFrame(
        rows
    )


print(
    "\nScoring development questions..."
)

development_df = score_questions(
    development
)


print(
    "Scoring held-out test questions..."
)

test_df = score_questions(
    test
)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    results,
    threshold
):

    # Similarity above threshold means:
    # allow answer / do not abstain.
    system_answers = (
        results[
            "max_similarity"
        ]
        >= threshold
    )

    answerable_mask = (
        results["label"]
        == "answerable"
    )

    unanswerable_mask = (
        results["label"]
        == "unanswerable"
    )

    # Answerable but system abstains.
    false_abstention_mask = (
        answerable_mask
        &
        (~system_answers)
    )

    # Unanswerable but system allows an answer.
    unsupported_answer_mask = (
        unanswerable_mask
        &
        system_answers
    )

    n_answerable = int(
        answerable_mask.sum()
    )

    n_unanswerable = int(
        unanswerable_mask.sum()
    )

    false_abstention_rate = (
        float(
            false_abstention_mask.sum()
        )
        /
        n_answerable
    )

    unsupported_answer_rate = (
        float(
            unsupported_answer_mask.sum()
        )
        /
        n_unanswerable
    )

    # Equal weighting of the two error types
    # for threshold selection.
    balanced_error = (
        false_abstention_rate
        +
        unsupported_answer_rate
    ) / 2.0

    return {
        "threshold":
            float(threshold),

        "false_abstention_rate":
            false_abstention_rate,

        "unsupported_answer_rate":
            unsupported_answer_rate,

        "balanced_error":
            balanced_error
    }


# ============================================================
# TUNE THRESHOLD ON DEVELOPMENT ONLY
# ============================================================

print("\n==============================")
print("DEVELOPMENT THRESHOLD TUNING")
print("==============================")


threshold_rows = []

for threshold in THRESHOLDS:

    threshold_rows.append(
        evaluate_threshold(
            development_df,
            threshold
        )
    )


threshold_df = pd.DataFrame(
    threshold_rows
)


best_row = threshold_df.loc[
    threshold_df[
        "balanced_error"
    ].idxmin()
]


selected_threshold = float(
    best_row["threshold"]
)


print(
    "Selected threshold:",
    selected_threshold
)

print(
    "Development false-abstention rate:",
    round(
        float(
            best_row[
                "false_abstention_rate"
            ]
        ),
        4
    )
)

print(
    "Development unsupported-answer rate:",
    round(
        float(
            best_row[
                "unsupported_answer_rate"
            ]
        ),
        4
    )
)

print(
    "Development balanced error:",
    round(
        float(
            best_row[
                "balanced_error"
            ]
        ),
        4
    )
)


# ============================================================
# FREEZE THRESHOLD AND APPLY TO HELD-OUT TEST
# ============================================================

test_df["decision"] = np.where(
    test_df[
        "max_similarity"
    ]
    >= selected_threshold,
    "answer",
    "abstain"
)


test_metrics = evaluate_threshold(
    test_df,
    selected_threshold
)


# ============================================================
# HELD-OUT TEST RESULTS
# ============================================================

print("\n==============================")
print("HELD-OUT ABSTENTION RESULTS")
print("==============================")


print(
    "Threshold:",
    selected_threshold
)

print(
    "False-abstention rate:",
    round(
        test_metrics[
            "false_abstention_rate"
        ],
        4
    )
)

print(
    "Unsupported-answer rate:",
    round(
        test_metrics[
            "unsupported_answer_rate"
        ],
        4
    )
)


# ============================================================
# CORRECT 95% CIs
#
# Each CI is calculated only over the relevant class:
#
# False-abstention:
#   15 answerable test questions
#
# Unsupported-answer:
#   15 unanswerable test questions
# ============================================================

answerable_test = test_df[
    test_df["label"]
    == "answerable"
].copy()

unanswerable_test = test_df[
    test_df["label"]
    == "unanswerable"
].copy()


false_abstention_values = (
    (
        answerable_test[
            "decision"
        ]
        == "abstain"
    )
    .astype(int)
    .to_numpy()
)


unsupported_answer_values = (
    (
        unanswerable_test[
            "decision"
        ]
        == "answer"
    )
    .astype(int)
    .to_numpy()
)


fa_low, fa_high = bootstrap_ci(
    false_abstention_values,
    seed=SEED + 1
)


ua_low, ua_high = bootstrap_ci(
    unsupported_answer_values,
    seed=SEED + 2
)


print(
    "False-abstention 95% CI:",
    f"[{fa_low:.4f}, {fa_high:.4f}]"
)

print(
    "Unsupported-answer 95% CI:",
    f"[{ua_low:.4f}, {ua_high:.4f}]"
)


# ============================================================
# ERROR COUNTS
# ============================================================

false_abstention_count = int(
    false_abstention_values.sum()
)

unsupported_answer_count = int(
    unsupported_answer_values.sum()
)


print(
    "\nFalse-abstentions:",
    false_abstention_count,
    "/",
    len(answerable_test)
)

print(
    "Unsupported answers:",
    unsupported_answer_count,
    "/",
    len(unanswerable_test)
)


# ============================================================
# ERROR EXAMPLES
# ============================================================

false_abstention_examples = test_df[
    (
        test_df["label"]
        == "answerable"
    )
    &
    (
        test_df["decision"]
        == "abstain"
    )
].copy()


unsupported_answer_examples = test_df[
    (
        test_df["label"]
        == "unanswerable"
    )
    &
    (
        test_df["decision"]
        == "answer"
    )
].copy()


print("\n==============================")
print("FALSE-ABSTENTION EXAMPLES")
print("==============================")


if len(false_abstention_examples) == 0:

    print(
        "None."
    )

else:

    print(
        false_abstention_examples[
            [
                "id",
                "chapter",
                "question",
                "max_similarity",
                "top1_section",
                "decision"
            ]
        ]
        .head(5)
        .to_string(
            index=False
        )
    )


print("\n==============================")
print("UNSUPPORTED-ANSWER EXAMPLES")
print("==============================")


if len(unsupported_answer_examples) == 0:

    print(
        "None."
    )

else:

    print(
        unsupported_answer_examples[
            [
                "id",
                "chapter",
                "question",
                "max_similarity",
                "top1_section",
                "decision"
            ]
        ]
        .head(5)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE DEVELOPMENT RESULTS
# ============================================================

development_df.to_csv(
    "abstention_development_results.csv",
    index=False
)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

test_df.to_csv(
    "abstention_test_results.csv",
    index=False
)


# ============================================================
# SAVE ALL THRESHOLD TUNING RESULTS
# ============================================================

threshold_df.to_csv(
    "abstention_threshold_tuning.csv",
    index=False
)


# ============================================================
# SAVE ERROR EXAMPLES
# ============================================================

false_abstention_examples.to_csv(
    "abstention_false_abstention_examples.csv",
    index=False
)


unsupported_answer_examples.to_csv(
    "abstention_unsupported_answer_examples.csv",
    index=False
)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {
    "dataset":
        DATASET,

    "question_file":
        QUESTIONS_FILE,

    "model":
        MODEL_NAME,

    "random_seed":
        SEED,

    "retrieval_top_k":
        TOP_K,

    "threshold_candidates":
        THRESHOLDS.tolist(),

    "development_size":
        len(development_df),

    "test_size":
        len(test_df),

    "development_answerable":
        int(
            (
                development_df[
                    "label"
                ]
                == "answerable"
            ).sum()
        ),

    "development_unanswerable":
        int(
            (
                development_df[
                    "label"
                ]
                == "unanswerable"
            ).sum()
        ),

    "test_answerable":
        int(
            (
                test_df[
                    "label"
                ]
                == "answerable"
            ).sum()
        ),

    "test_unanswerable":
        int(
            (
                test_df[
                    "label"
                ]
                == "unanswerable"
            ).sum()
        ),

    "selected_threshold":
        selected_threshold,

    "test_false_abstention_rate":
        float(
            test_metrics[
                "false_abstention_rate"
            ]
        ),

    "test_false_abstention_ci_95":
        [
            fa_low,
            fa_high
        ],

    "test_unsupported_answer_rate":
        float(
            test_metrics[
                "unsupported_answer_rate"
            ]
        ),

    "test_unsupported_answer_ci_95":
        [
            ua_low,
            ua_high
        ],

    "false_abstention_count":
        false_abstention_count,

    "unsupported_answer_count":
        unsupported_answer_count
}


with open(
    "abstention_benchmark_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print("\n==============================")
print("DONE")
print("==============================")

print(
    "Saved: abstention_benchmark_results.json"
)

print(
    "Saved: abstention_development_results.csv"
)

print(
    "Saved: abstention_test_results.csv"
)

print(
    "Saved: abstention_threshold_tuning.csv"
)

print(
    "Saved: abstention_false_abstention_examples.csv"
)

print(
    "Saved: abstention_unsupported_answer_examples.csv"
)