import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

DATASET_FILE = "Merged_Chapter_Dataset_recovered.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5

OUTPUT_RESULTS = "qa_pair_retrieval_486_results.csv"
OUTPUT_SUMMARY = "qa_pair_retrieval_486_summary.json"


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    """
    Convert a value to clean text.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def parse_list(value):
    """
    Parse cells that contain Python-style lists.
    Examples:
        "['q1', 'q2']"
        "['a1', 'a2']"
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [
                clean_text(x)
                for x in parsed
            ]

    except Exception:
        pass

    return []


def find_column(columns, candidates):
    """
    Find a column using case-insensitive exact matching,
    then partial matching.
    """
    normalized = {
        str(c).strip().lower(): c
        for c in columns
    }

    # Exact match
    for candidate in candidates:

        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    # Partial match
    for candidate in candidates:

        key = candidate.strip().lower()

        for normalized_name, original in normalized.items():

            if key in normalized_name:
                return original

    return None


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("==============================")
print("LOADING DATASET")
print("==============================")

if not Path(DATASET_FILE).exists():

    raise FileNotFoundError(
        f"Could not find {DATASET_FILE}"
    )


df = pd.read_csv(
    DATASET_FILE
)

print(
    "Rows:",
    len(df)
)

print()
print("Columns:")

for col in df.columns:
    print(
        " -",
        col
    )


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

chapter_col = find_column(
    df.columns,
    [
        "chapter",
        "chapter_name",
        "section",
        "section_name"
    ]
)

question_col = find_column(
    df.columns,
    [
        "question",
        "questions",
        "chapter_questions",
        "generated_questions"
    ]
)

answer_col = find_column(
    df.columns,
    [
        "answer",
        "answers",
        "chapter_answers",
        "generated_answers"
    ]
)


print()
print("==============================")
print("COLUMN DETECTION")
print("==============================")

print(
    "Chapter column:",
    chapter_col
)

print(
    "Question column:",
    question_col
)

print(
    "Answer column:",
    answer_col
)


if question_col is None:

    raise RuntimeError(
        "Could not identify the question column."
    )

if answer_col is None:

    raise RuntimeError(
        "Could not identify the answer column."
    )


# ============================================================
# BUILD INDIVIDUAL Q-A RECORDS
# ============================================================

print()
print("==============================")
print("BUILDING Q-A PAIRS")
print("==============================")


qa_records = []

pair_counter = 0


for row_index, row in df.iterrows():

    chapter = ""

    if chapter_col is not None:
        chapter = clean_text(
            row[chapter_col]
        )

    questions = parse_list(
        row[question_col]
    )

    answers = parse_list(
        row[answer_col]
    )


    # --------------------------------------------------------
    # CASE 1:
    # Questions and answers are stored as lists
    # --------------------------------------------------------

    if questions:

        for position, question in enumerate(
            questions
        ):

            if not question:
                continue

            if position < len(answers):

                answer = answers[position]

            else:

                answer = ""

            pair_counter += 1

            qa_records.append({
                "pair_id":
                    f"QA{pair_counter:03d}",

                "source_row":
                    row_index,

                "pair_position":
                    position + 1,

                "chapter":
                    chapter,

                "question":
                    question,

                "answer":
                    answer
            })


    # --------------------------------------------------------
    # CASE 2:
    # Dataset contains one question/answer per row
    # --------------------------------------------------------

    elif (
        clean_text(row[question_col])
        and clean_text(row[answer_col])
    ):

        pair_counter += 1

        qa_records.append({
            "pair_id":
                f"QA{pair_counter:03d}",

            "source_row":
                row_index,

            "pair_position":
                1,

            "chapter":
                chapter,

            "question":
                clean_text(
                    row[question_col]
                ),

            "answer":
                clean_text(
                    row[answer_col]
                )
        })


qa = pd.DataFrame(
    qa_records
)


print(
    "Prepared Q-A pairs:",
    len(qa)
)


# ============================================================
# CHECK EXPECTED SCALE
# ============================================================

if len(qa) != 486:

    print()
    print(
        "WARNING:"
    )

    print(
        "Expected approximately 486 "
        "prepared Q-A pairs."
    )

    print(
        "Detected:",
        len(qa)
    )

    print(
        "The script will continue, but verify "
        "the detected question/answer columns."
    )


# ============================================================
# REMOVE EMPTY PAIRS
# ============================================================

qa = qa[
    (
        qa["question"]
        .astype(str)
        .str.strip()
        != ""
    )
].copy()


qa = qa.reset_index(
    drop=True
)


# ============================================================
# BUILD RETRIEVAL DOCUMENT
#
# We intentionally retrieve the PREPARED Q-A PAIRS.
#
# Each document contains:
#   question + answer
#
# This tests whether a prepared learner question
# recovers its own prepared instructional pair.
# ============================================================

qa["retrieval_document"] = (
    "Question: "
    + qa["question"].astype(str)
    + "\nAnswer: "
    + qa["answer"].astype(str)
)


documents = qa[
    "retrieval_document"
].tolist()


# ============================================================
# LOAD MINILM
# ============================================================

print()
print("==============================")
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
# ENCODE ALL Q-A DOCUMENTS
# ============================================================

print()
print("==============================")
print("ENCODING Q-A DOCUMENTS")
print("==============================")


document_embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# RETRIEVE
# ============================================================

print()
print("==============================")
print("RUNNING 486-PAIR RETRIEVAL")
print("==============================")


results = []


for i, row in qa.iterrows():

    question = row["question"]

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]


    scores = (
        document_embeddings
        @ query_embedding
    )


    ranking = np.argsort(
        -scores
    )


    top_indices = ranking[
        :TOP_K
    ]


    # --------------------------------------------------------
    # Find rank of the TRUE originating pair
    # --------------------------------------------------------

    true_index = i

    matching_positions = np.where(
        ranking == true_index
    )[0]


    if len(matching_positions) == 0:

        true_rank = None

    else:

        true_rank = (
            int(
                matching_positions[0]
            )
            + 1
        )


    # --------------------------------------------------------
    # Top-k IDs / chapters
    # --------------------------------------------------------

    top_ids = []

    top_chapters = []

    top_scores = []


    for idx in top_indices:

        idx = int(idx)

        top_ids.append(
            qa.iloc[idx]["pair_id"]
        )

        top_chapters.append(
            qa.iloc[idx]["chapter"]
        )

        top_scores.append(
            float(
                scores[idx]
            )
        )


    results.append({

        "pair_id":
            row["pair_id"],

        "source_row":
            row["source_row"],

        "pair_position":
            row["pair_position"],

        "chapter":
            row["chapter"],

        "question":
            row["question"],

        "reference_answer":
            row["answer"],

        "true_rank":
            true_rank,

        "true_pair_similarity":
            float(
                scores[true_index]
            ),

        "top1_pair_id":
            top_ids[0],

        "top1_chapter":
            top_chapters[0],

        "top1_score":
            top_scores[0],

        "top2_pair_id":
            top_ids[1],

        "top2_chapter":
            top_chapters[1],

        "top2_score":
            top_scores[1],

        "top3_pair_id":
            top_ids[2],

        "top3_chapter":
            top_chapters[2],

        "top3_score":
            top_scores[2],

        "top4_pair_id":
            top_ids[3],

        "top4_chapter":
            top_chapters[3],

        "top4_score":
            top_scores[3],

        "top5_pair_id":
            top_ids[4],

        "top5_chapter":
            top_chapters[4],

        "top5_score":
            top_scores[4]
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# METRICS
# ============================================================

print()
print("==============================")
print("CALCULATING METRICS")
print("==============================")


n = len(
    results_df
)


top1_hits = int(
    (
        results_df[
            "true_rank"
        ]
        <= 1
    ).sum()
)


top3_hits = int(
    (
        results_df[
            "true_rank"
        ]
        <= 3
    ).sum()
)


top5_hits = int(
    (
        results_df[
            "true_rank"
        ]
        <= 5
    ).sum()
)


top1_recall = (
    top1_hits
    / n
)


top3_recall = (
    top3_hits
    / n
)


top5_recall = (
    top5_hits
    / n
)


# ============================================================
# CHAPTER-LEVEL RECOVERY
# ============================================================

chapter_top1_hits = int(
    (
        results_df[
            "chapter"
        ].astype(str).str.strip()
        ==
        results_df[
            "top1_chapter"
        ].astype(str).str.strip()
    ).sum()
)


chapter_top1_accuracy = (
    chapter_top1_hits
    / n
)


# ============================================================
# MEAN RECIPROCAL RANK
# ============================================================

def reciprocal_rank(rank):

    if pd.isna(rank):
        return 0.0

    return (
        1.0
        / float(rank)
    )


mrr = float(
    results_df[
        "true_rank"
    ]
    .apply(
        reciprocal_rank
    )
    .mean()
)


# ============================================================
# MEDIAN TRUE RANK
# ============================================================

median_true_rank = float(
    results_df[
        "true_rank"
    ].median()
)


# ============================================================
# TOP-1 SIMILARITY
# ============================================================

mean_top1_score = float(
    results_df[
        "top1_score"
    ].mean()
)


mean_true_pair_score = float(
    results_df[
        "true_pair_similarity"
    ].mean()
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "dataset":
        DATASET_FILE,

    "number_of_pairs":
        int(n),

    "embedding_model":
        MODEL_NAME,

    "retrieval_document":
        "prepared question + prepared answer",

    "top_k":
        TOP_K,

    "top1_hits":
        top1_hits,

    "top1_recall":
        float(top1_recall),

    "top3_hits":
        top3_hits,

    "top3_recall":
        float(top3_recall),

    "top5_hits":
        top5_hits,

    "top5_recall":
        float(top5_recall),

    "chapter_top1_hits":
        chapter_top1_hits,

    "chapter_top1_accuracy":
        float(chapter_top1_accuracy),

    "mrr":
        mrr,

    "median_true_rank":
        median_true_rank,

    "mean_top1_similarity":
        mean_top1_score,

    "mean_true_pair_similarity":
        mean_true_pair_score
}


with open(
    OUTPUT_SUMMARY,
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
print("486 Q-A PAIR RETRIEVAL RESULTS")
print("==============================")


print(
    "Pairs evaluated:",
    n
)


print(
    f"Top-1 pair recovery: "
    f"{top1_hits}/{n} "
    f"({top1_recall:.4%})"
)


print(
    f"Recall@3: "
    f"{top3_hits}/{n} "
    f"({top3_recall:.4%})"
)


print(
    f"Recall@5: "
    f"{top5_hits}/{n} "
    f"({top5_recall:.4%})"
)


print(
    f"Chapter Top-1 accuracy: "
    f"{chapter_top1_hits}/{n} "
    f"({chapter_top1_accuracy:.4%})"
)


print(
    f"MRR: "
    f"{mrr:.4f}"
)


print(
    f"Median true-pair rank: "
    f"{median_true_rank:.1f}"
)


print(
    f"Mean top-1 similarity: "
    f"{mean_top1_score:.4f}"
)


print(
    f"Mean true-pair similarity: "
    f"{mean_true_pair_score:.4f}"
)


# ============================================================
# SHOW EXAMPLES OF MISSED PAIRS
# ============================================================

misses = results_df[
    results_df[
        "true_rank"
    ]
    > 5
].copy()


print()
print("==============================")
print("TOP RETRIEVAL FAILURES")
print("==============================")


if len(misses) == 0:

    print(
        "No failures beyond rank 5."
    )

else:

    print(
        misses[
            [
                "pair_id",
                "chapter",
                "question",
                "true_rank",
                "top1_pair_id",
                "top1_chapter",
                "top1_score"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


print()
print("==============================")
print("FILES SAVED")
print("==============================")


print(
    OUTPUT_RESULTS
)

print(
    OUTPUT_SUMMARY
)


print()
print("DONE")