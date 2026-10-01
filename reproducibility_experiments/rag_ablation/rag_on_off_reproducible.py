import ast
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# REPRODUCIBILITY SETTINGS
# ============================================================

SEED = 42
N_QUESTIONS = 40
TOP_K = 3

DATASET_FILE = "Merged_Chapter_Dataset_recovered.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "tiiuae/falcon-rw-1b"

OUTPUT_SAMPLE = "rag_ablation_sample.csv"
OUTPUT_BLINDED = "rag_ablation_blinded_review.csv"
OUTPUT_KEY = "rag_ablation_condition_key.csv"
OUTPUT_METADATA = "rag_ablation_metadata.json"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip()
    )


def parse_list(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [
            clean_text(x)
            for x in value
        ]

    text = str(value).strip()

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

    normalized = {
        str(c).strip().lower(): c
        for c in columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in normalized:
            return normalized[key]

    for candidate in candidates:

        key = candidate.lower()

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
    raise FileNotFoundError(DATASET_FILE)


df = pd.read_csv(DATASET_FILE)


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
        "questions",
        "question",
        "chapter_questions",
        "generated_questions"
    ]
)


answer_col = find_column(
    df.columns,
    [
        "answers",
        "answer",
        "chapter_answers",
        "generated_answers"
    ]
)


content_col = find_column(
    df.columns,
    [
        "content",
        "chapter_content",
        "source_content",
        "text",
        "chapter_text"
    ]
)


summary_col = find_column(
    df.columns,
    [
        "summary",
        "chapter_summary",
        "summaries"
    ]
)


print("Rows:", len(df))
print("Chapter column:", chapter_col)
print("Question column:", question_col)
print("Answer column:", answer_col)
print("Content column:", content_col)
print("Summary column:", summary_col)


if question_col is None:
    raise RuntimeError(
        "Could not identify question column."
    )


if answer_col is None:
    raise RuntimeError(
        "Could not identify answer column."
    )


# ============================================================
# BUILD THE 486 Q-A PAIRS
# ============================================================

records = []

pair_number = 0


for source_row, row in df.iterrows():

    chapter = (
        clean_text(row[chapter_col])
        if chapter_col is not None
        else ""
    )

    questions = parse_list(
        row[question_col]
    )

    answers = parse_list(
        row[answer_col]
    )

    source_content = (
        clean_text(row[content_col])
        if content_col is not None
        else ""
    )

    summary = (
        clean_text(row[summary_col])
        if summary_col is not None
        else ""
    )


    if questions:

        for position, question in enumerate(questions):

            if not question:
                continue

            answer = (
                answers[position]
                if position < len(answers)
                else ""
            )

            pair_number += 1

            records.append({

                "pair_id":
                    f"QA{pair_number:03d}",

                "source_row":
                    source_row,

                "pair_position":
                    position + 1,

                "chapter":
                    chapter,

                "question":
                    question,

                "reference_answer":
                    answer,

                "source_content":
                    source_content,

                "summary":
                    summary
            })


qa = pd.DataFrame(records)


print()
print("Prepared Q-A pairs:", len(qa))


if len(qa) != 486:

    print(
        "WARNING: Expected 486 pairs, "
        f"but detected {len(qa)}."
    )


# ============================================================
# FIXED-SEED SAMPLE OF 40
#
# IMPORTANT:
# Sampling occurs BEFORE model generation and scoring.
# ============================================================

sample = qa.sample(
    n=N_QUESTIONS,
    random_state=SEED,
    replace=False
).copy()


sample = sample.sort_values(
    "pair_id"
).reset_index(drop=True)


sample["experiment_id"] = [
    f"E{i:02d}"
    for i in range(
        1,
        len(sample) + 1
    )
]


sample.to_csv(
    OUTPUT_SAMPLE,
    index=False,
    encoding="utf-8-sig"
)


print()
print("==============================")
print("FIXED EVALUATION SAMPLE")
print("==============================")


print(
    f"Selected {len(sample)} questions "
    f"with random seed {SEED}."
)

print(
    "Sample fixed before response generation."
)


# ============================================================
# BUILD RETRIEVAL CORPUS
#
# Prepared question + prepared answer.
# This corresponds to the prepared instructional
# resources used by the lightweight QA path.
# ============================================================

qa["retrieval_document"] = (
    "Chapter: "
    + qa["chapter"].astype(str)
    + "\nQuestion: "
    + qa["question"].astype(str)
    + "\nAnswer: "
    + qa["reference_answer"].astype(str)
)


retrieval_documents = (
    qa["retrieval_document"]
    .tolist()
)


# ============================================================
# LOAD MINILM
# ============================================================

print()
print("==============================")
print("LOADING RETRIEVER")
print("==============================")


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


document_embeddings = embedding_model.encode(
    retrieval_documents,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# LOAD FALCON
# ============================================================

print()
print("==============================")
print("LOADING GENERATION MODEL")
print("==============================")


tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL
)


model = AutoModelForCausalLM.from_pretrained(
    GENERATION_MODEL
)


model.eval()


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# DETERMINISTIC GENERATION
#
# Same model/settings in both conditions.
#
# We deliberately use do_sample=False so the only
# intended experimental difference is retrieved context.
# ============================================================

MAX_INPUT_TOKENS = 768
MAX_NEW_TOKENS = 160


def generate(prompt):

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )


    with torch.no_grad():

        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )


    generated = output[
        0,
        encoded["input_ids"].shape[1]:
    ]


    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(question, k=TOP_K):

    query_embedding = embedding_model.encode(
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
    )[:k]


    retrieved = []


    for rank, idx in enumerate(
        ranking,
        start=1
    ):

        idx = int(idx)

        retrieved.append({

            "rank":
                rank,

            "pair_id":
                qa.iloc[idx]["pair_id"],

            "chapter":
                qa.iloc[idx]["chapter"],

            "question":
                qa.iloc[idx]["question"],

            "answer":
                qa.iloc[idx]["reference_answer"],

            "similarity":
                float(scores[idx])
        })


    return retrieved


# ============================================================
# PROMPTS
#
# Keep instructional wording as similar as possible.
# Only retrieved evidence differs.
# ============================================================

BASE_INSTRUCTION = """
You are an educational tutor.
Answer the learner's question concisely and clearly.
Do not invent information.
""".strip()


def make_rag_on_prompt(question, retrieved):

    context_parts = []

    for item in retrieved:

        context_parts.append(
            f"[Retrieved {item['rank']}]\n"
            f"Chapter: {item['chapter']}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )


    context = "\n\n".join(
        context_parts
    )


    return f"""
{BASE_INSTRUCTION}

Use the retrieved instructional context below when answering.

RETRIEVED CONTEXT:
{context}

LEARNER QUESTION:
{question}

ANSWER:
""".strip()


def make_rag_off_prompt(question):

    return f"""
{BASE_INSTRUCTION}

LEARNER QUESTION:
{question}

ANSWER:
""".strip()


# ============================================================
# GENERATE PAIRED RESPONSES
# ============================================================

print()
print("==============================")
print("GENERATING PAIRED RESPONSES")
print("==============================")


generated_records = []


for i, row in sample.iterrows():

    experiment_id = row["experiment_id"]

    question = row["question"]


    print(
        f"[{i + 1:02d}/{len(sample)}] "
        f"{experiment_id}"
    )


    retrieved = retrieve(
        question,
        TOP_K
    )


    on_prompt = make_rag_on_prompt(
        question,
        retrieved
    )


    off_prompt = make_rag_off_prompt(
        question
    )


    response_on = generate(
        on_prompt
    )


    response_off = generate(
        off_prompt
    )


    generated_records.append({

        "experiment_id":
            experiment_id,

        "pair_id":
            row["pair_id"],

        "chapter":
            row["chapter"],

        "question":
            question,

        "reference_answer":
            row["reference_answer"],

        "retrieved_context":
            "\n\n".join([
                (
                    f"RANK {x['rank']} | "
                    f"{x['chapter']} | "
                    f"score={x['similarity']:.4f}\n"
                    f"Q: {x['question']}\n"
                    f"A: {x['answer']}"
                )
                for x in retrieved
            ]),

        "rag_on_response":
            response_on,

        "rag_off_response":
            response_off
    })


generated_df = pd.DataFrame(
    generated_records
)


# ============================================================
# BLIND THE CONDITIONS
#
# A/B assignment is deterministic from SEED but hidden
# from the human-review file.
# ============================================================

blind_rng = random.Random(
    SEED + 1000
)


review_rows = []

key_rows = []


for _, row in generated_df.iterrows():

    on_is_a = blind_rng.choice(
        [True, False]
    )


    if on_is_a:

        response_a = row["rag_on_response"]
        response_b = row["rag_off_response"]

        condition_a = "RAG_ON"
        condition_b = "RAG_OFF"

    else:

        response_a = row["rag_off_response"]
        response_b = row["rag_on_response"]

        condition_a = "RAG_OFF"
        condition_b = "RAG_ON"


    review_rows.append({

        "experiment_id":
            row["experiment_id"],

        "chapter":
            row["chapter"],

        "question":
            row["question"],

        "reference_answer":
            row["reference_answer"],

        "retrieved_context":
            row["retrieved_context"],

        "response_A":
            response_a,

        "response_B":
            response_b,

        # ------------------------------
        # HUMAN SCORES
        # ------------------------------

        "A_factuality_0_2":
            "",

        "A_grounding_0_2":
            "",

        "A_clarity_0_1":
            "",

        "B_factuality_0_2":
            "",

        "B_grounding_0_2":
            "",

        "B_clarity_0_1":
            "",

        "review_notes":
            ""
    })


    key_rows.append({

        "experiment_id":
            row["experiment_id"],

        "pair_id":
            row["pair_id"],

        "condition_A":
            condition_a,

        "condition_B":
            condition_b
    })


review_df = pd.DataFrame(
    review_rows
)


key_df = pd.DataFrame(
    key_rows
)


review_df.to_csv(
    OUTPUT_BLINDED,
    index=False,
    encoding="utf-8-sig"
)


key_df.to_csv(
    OUTPUT_KEY,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE EXPERIMENT METADATA
# ============================================================

metadata = {

    "dataset":
        DATASET_FILE,

    "number_of_available_pairs":
        int(len(qa)),

    "number_of_evaluation_questions":
        N_QUESTIONS,

    "sampling_method":
        "simple random sample without replacement",

    "sampling_seed":
        SEED,

    "sample_fixed_before_generation":
        True,

    "embedding_model":
        EMBEDDING_MODEL,

    "generation_model":
        GENERATION_MODEL,

    "retrieval_k":
        TOP_K,

    "rag_on":
        "top-3 prepared instructional Q-A resources supplied",

    "rag_off":
        "no retrieved instructional context supplied",

    "generation_do_sample":
        False,

    "generation_max_input_tokens":
        MAX_INPUT_TOKENS,

    "generation_max_new_tokens":
        MAX_NEW_TOKENS,

    "paired_design":
        True,

    "blinded_review":
        True,

    "blinding_method":
        "RAG ON/OFF randomly mapped to Response A/B using fixed seed",

    "human_raters":
        1,

    "rubric": {

        "factuality":
            "0-2",

        "grounding":
            "0-2",

        "clarity":
            "0-1",

        "maximum_total":
            5
    },

    "planned_primary_analysis":
        "paired comparison of total rubric scores",

    "planned_test":
        "Wilcoxon signed-rank test",

    "planned_confidence_interval":
        "95% bootstrap CI for paired mean score difference",

    "bootstrap_resamples":
        10000
}


with open(
    OUTPUT_METADATA,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print()
print("==============================")
print("EXPERIMENT GENERATION COMPLETE")
print("==============================")


print(
    "Evaluation questions:",
    len(sample)
)

print(
    "Generation model:",
    GENERATION_MODEL
)

print(
    "Embedding model:",
    EMBEDDING_MODEL
)

print(
    "Retrieval k:",
    TOP_K
)

print(
    "Sampling seed:",
    SEED
)

print()
print("Saved:")
print(" ", OUTPUT_SAMPLE)
print(" ", OUTPUT_BLINDED)
print(" ", OUTPUT_KEY)
print(" ", OUTPUT_METADATA)

print()
print(
    "IMPORTANT: Do NOT open "
    "rag_ablation_condition_key.csv before scoring."
)

print(
    "Score only rag_ablation_blinded_review.csv."
)

print()
print("NEXT:")
print(
    "Complete factuality, grounding, and clarity "
    "scores for Response A and Response B."
)