import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = "Merged_Chapter_Dataset.csv"
OUTPUT_FILE = "independent_learner_questions_to_write.csv"

SEED = 42
QUESTIONS_PER_TOP_CHAPTER = 3


# ============================================================
# LOAD
# ============================================================

print()
print("==============================")
print("INDEPENDENT QUESTION SAMPLING")
print("==============================")

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}. "
        "Change INPUT_FILE to your actual merged dataset filename."
    )

df = pd.read_csv(INPUT_FILE)

required = [
    "chapter",
    "Chapter Content",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing)
    )

print("Rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# REMOVE DUPLICATE SECTION ROWS
# ============================================================

df = df.copy()

df["chapter"] = df["chapter"].astype(str).str.strip()

df = df.drop_duplicates(
    subset=["chapter"],
    keep="first"
).reset_index(drop=True)


# ============================================================
# DERIVE TOP-LEVEL CHAPTER
#
# Example:
# 1.1 Computing from Inception to Today -> 1
# 1.2 Computer Hardware and Networks     -> 1
# 2.3 Communication ...                   -> 2
# ============================================================

def top_chapter(section_name):

    text = str(section_name).strip()

    if "." in text:
        prefix = text.split(".", 1)[0].strip()

        if prefix.isdigit():
            return int(prefix)

    return None


df["top_chapter"] = df["chapter"].apply(
    top_chapter
)

if df["top_chapter"].isna().any():
    bad = df.loc[
        df["top_chapter"].isna(),
        "chapter"
    ].tolist()

    raise RuntimeError(
        "Could not determine top-level chapter for:\n"
        + "\n".join(bad)
    )


# ============================================================
# CHECK CHAPTER COUNT
# ============================================================

chapters = sorted(
    df["top_chapter"].unique()
)

print(
    "Top-level chapters found:",
    chapters
)

print(
    "Number of top-level chapters:",
    len(chapters)
)


# ============================================================
# SAMPLE 3 SECTIONS PER TOP-LEVEL CHAPTER
#
# This produces a balanced 45-question target:
# 15 chapters x 3 sections = 45
# ============================================================

rng = np.random.default_rng(SEED)

sampled = []

for ch in chapters:

    group = df[
        df["top_chapter"] == ch
    ].copy()

    n = min(
        QUESTIONS_PER_TOP_CHAPTER,
        len(group)
    )

    if n < QUESTIONS_PER_TOP_CHAPTER:
        print(
            f"Warning: top-level chapter {ch} "
            f"contains only {len(group)} sections."
        )

    indices = rng.choice(
        len(group),
        size=n,
        replace=False
    )

    chosen = group.iloc[
        indices
    ].copy()

    sampled.append(chosen)


sampled_df = pd.concat(
    sampled,
    ignore_index=True
)


# ============================================================
# CREATE HUMAN-AUTHORING SHEET
#
# IMPORTANT:
# We deliberately DO NOT export Questions or Answers.
# This prevents the generated QA pairs from being shown
# to the person writing the independent learner questions.
# ============================================================

authoring = pd.DataFrame({
    "question_id": [
        f"IQ{i+1:03d}"
        for i in range(len(sampled_df))
    ],

    "top_chapter":
        sampled_df["top_chapter"].astype(int),

    "source_section":
        sampled_df["chapter"],

    "source_text":
        sampled_df["Chapter Content"].fillna(""),

    "independent_learner_question":
        "",

    "question_author_notes":
        "",
})


# ============================================================
# SORT BY CHAPTER
# ============================================================

authoring = authoring.sort_values(
    [
        "top_chapter",
        "source_section"
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

authoring.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# REPORT
# ============================================================

print()
print("==============================")
print("SAMPLING COMPLETE")
print("==============================")

print(
    "Independent-question targets:",
    len(authoring)
)

print(
    "Questions per top-level chapter:",
    QUESTIONS_PER_TOP_CHAPTER
)

print()
print("Distribution:")

distribution = (
    authoring
    .groupby("top_chapter")
    .size()
    .to_string()
)

print(distribution)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("==============================")
print("NEXT")
print("==============================")
print(
    "Open the CSV and manually write ONE "
    "natural learner-style question for each row."
)

print(
    "Do NOT copy, paraphrase, or consult the "
    "486 generated Q-A pairs while writing."
)

print(
    "Keep the source_section and source_text unchanged."
)