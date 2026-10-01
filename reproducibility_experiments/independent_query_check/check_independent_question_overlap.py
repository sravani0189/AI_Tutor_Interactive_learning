import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
import re

INDEPENDENT_FILE = "independent_learner_questions_filled.csv"
ORIGINAL_FILE = "Merged_Chapter_Dataset.csv"

THRESHOLDS = {
    "exact": 1.0,
    "high": 0.90,
    "moderate": 0.80,
}


def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


if not Path(INDEPENDENT_FILE).exists():
    raise FileNotFoundError(INDEPENDENT_FILE)

if not Path(ORIGINAL_FILE).exists():
    raise FileNotFoundError(ORIGINAL_FILE)


ind = pd.read_csv(INDEPENDENT_FILE)
orig = pd.read_csv(ORIGINAL_FILE)

required_ind = [
    "question_id",
    "independent_learner_question",
]

required_orig = [
    "Questions",
]

for c in required_ind:
    if c not in ind.columns:
        raise RuntimeError(f"Missing column in independent file: {c}")

for c in required_orig:
    if c not in orig.columns:
        raise RuntimeError(f"Missing column in original file: {c}")


ind_questions = (
    ind["independent_learner_question"]
    .fillna("")
    .astype(str)
    .map(normalize)
)

orig_questions = (
    orig["Questions"]
    .fillna("")
    .astype(str)
    .map(normalize)
)

# Remove blank original questions.
orig_questions = orig_questions[
    orig_questions != ""
].drop_duplicates()


print()
print("==============================")
print("INDEPENDENT QUESTION OVERLAP")
print("==============================")

print("Independent questions:", len(ind_questions))
print("Original generated questions:", len(orig_questions))


exact_matches = []
high_matches = []
moderate_matches = []


for qid, q in zip(
    ind["question_id"],
    ind_questions
):

    best_score = 0.0
    best_original = ""

    for oq in orig_questions:

        score = SequenceMatcher(
            None,
            q,
            oq
        ).ratio()

        if score > best_score:
            best_score = score
            best_original = oq

    record = {
        "question_id": qid,
        "independent_question": q,
        "best_original_match": best_original,
        "similarity": best_score,
    }

    if best_score == THRESHOLDS["exact"]:
        exact_matches.append(record)

    elif best_score >= THRESHOLDS["high"]:
        high_matches.append(record)

    elif best_score >= THRESHOLDS["moderate"]:
        moderate_matches.append(record)


print()
print("Exact matches:", len(exact_matches))
print("Similarity >= 0.90:", len(high_matches))
print("Similarity >= 0.80:", len(moderate_matches))


if high_matches:

    print()
    print("==============================")
    print("HIGH-OVERLAP QUESTIONS")
    print("==============================")

    for r in sorted(
        high_matches,
        key=lambda x: -x["similarity"]
    ):

        print()
        print(
            r["question_id"],
            f"{r['similarity']:.3f}"
        )

        print(
            "Independent:",
            r["independent_question"]
        )

        print(
            "Original:",
            r["best_original_match"]
        )


# Save detailed audit.
audit_records = []

for qid, q in zip(
    ind["question_id"],
    ind_questions
):

    best_score = 0.0
    best_original = ""

    for oq in orig_questions:

        score = SequenceMatcher(
            None,
            q,
            oq
        ).ratio()

        if score > best_score:
            best_score = score
            best_original = oq

    audit_records.append({
        "question_id": qid,
        "independent_question": q,
        "closest_original_question": best_original,
        "similarity": best_score,
    })


audit = pd.DataFrame(audit_records)

audit.to_csv(
    "independent_question_overlap_audit.csv",
    index=False,
    encoding="utf-8-sig"
)


print()
print("Saved:")
print("independent_question_overlap_audit.csv")

print()
print("==============================")
print("DONE")
print("==============================")