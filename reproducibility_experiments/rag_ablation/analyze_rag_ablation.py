import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# FILES
# ============================================================

REVIEW_FILE = "rag_ablation_blinded_review.csv"
KEY_FILE = "rag_ablation_condition_key.csv"

OUTPUT_RESULTS = "rag_ablation_final_results.csv"
OUTPUT_SUMMARY = "rag_ablation_final_summary.json"


# ============================================================
# SETTINGS
# ============================================================

BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10000

EXPECTED_N = 40

REQUIRED_REVIEW_COLUMNS = [
    "experiment_id",
    "question",
    "A_factuality_0_2",
    "A_grounding_0_2",
    "A_clarity_0_1",
    "B_factuality_0_2",
    "B_grounding_0_2",
    "B_clarity_0_1",
]

REQUIRED_KEY_COLUMNS = [
    "experiment_id",
    "condition_A",
    "condition_B",
]


# ============================================================
# HELPERS
# ============================================================

def require_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def numeric_column(df, column):
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    if df[column].isna().any():
        bad = df.loc[
            df[column].isna(),
            ["experiment_id", column]
        ]

        raise RuntimeError(
            f"Blank/non-numeric scores found in {column}:\n"
            f"{bad.to_string(index=False)}"
        )


def percentile_ci(values, seed=42, n_boot=10000):

    values = np.asarray(
        values,
        dtype=float
    )

    rng = np.random.default_rng(seed)

    n = len(values)

    boot_means = np.empty(
        n_boot,
        dtype=float
    )

    for i in range(n_boot):

        sample = rng.choice(
            values,
            size=n,
            replace=True
        )

        boot_means[i] = np.mean(
            sample
        )

    low, high = np.percentile(
        boot_means,
        [2.5, 97.5]
    )

    return float(low), float(high)


def paired_bootstrap_mean_difference(
    differences,
    seed=42,
    n_boot=10000
):

    differences = np.asarray(
        differences,
        dtype=float
    )

    rng = np.random.default_rng(seed)

    n = len(differences)

    boot_means = np.empty(
        n_boot,
        dtype=float
    )

    for i in range(n_boot):

        sample = rng.choice(
            differences,
            size=n,
            replace=True
        )

        boot_means[i] = np.mean(
            sample
        )

    low, high = np.percentile(
        boot_means,
        [2.5, 97.5]
    )

    return float(low), float(high)


def summarize(values):

    values = np.asarray(
        values,
        dtype=float
    )

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# ============================================================
# LOAD FILES
# ============================================================

print()
print("==============================")
print("RAG ON/OFF FINAL ANALYSIS")
print("==============================")


require_file(REVIEW_FILE)
require_file(KEY_FILE)


review = pd.read_csv(
    REVIEW_FILE
)

key = pd.read_csv(
    KEY_FILE
)


# ============================================================
# VALIDATE STRUCTURE
# ============================================================

missing_review = [
    c for c in REQUIRED_REVIEW_COLUMNS
    if c not in review.columns
]

missing_key = [
    c for c in REQUIRED_KEY_COLUMNS
    if c not in key.columns
]


if missing_review:
    raise RuntimeError(
        "Missing review columns: "
        + ", ".join(missing_review)
    )


if missing_key:
    raise RuntimeError(
        "Missing key columns: "
        + ", ".join(missing_key)
    )


print(
    "Review rows:",
    len(review)
)

print(
    "Key rows:",
    len(key)
)


if len(review) != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} review rows, "
        f"found {len(review)}."
    )


if len(key) != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} key rows, "
        f"found {len(key)}."
    )


# ============================================================
# CHECK IDS
# ============================================================

review_ids = set(
    review["experiment_id"]
)

key_ids = set(
    key["experiment_id"]
)


if review_ids != key_ids:

    only_review = sorted(
        review_ids - key_ids
    )

    only_key = sorted(
        key_ids - review_ids
    )

    raise RuntimeError(
        "Experiment IDs do not match.\n"
        f"Only in review: {only_review}\n"
        f"Only in key: {only_key}"
    )


# ============================================================
# NUMERIC VALIDATION
# ============================================================

score_columns = [
    "A_factuality_0_2",
    "A_grounding_0_2",
    "A_clarity_0_1",
    "B_factuality_0_2",
    "B_grounding_0_2",
    "B_clarity_0_1",
]


for col in score_columns:
    numeric_column(
        review,
        col
    )


# Range checks

for col in [
    "A_factuality_0_2",
    "B_factuality_0_2",
]:

    if not review[col].between(
        0, 2
    ).all():

        raise RuntimeError(
            f"Invalid value in {col}; "
            "expected 0-2."
        )


for col in [
    "A_grounding_0_2",
    "B_grounding_0_2",
]:

    if not review[col].between(
        0, 2
    ).all():

        raise RuntimeError(
            f"Invalid value in {col}; "
            "expected 0-2."
        )


for col in [
    "A_clarity_0_1",
    "B_clarity_0_1",
]:

    if not review[col].between(
        0, 1
    ).all():

        raise RuntimeError(
            f"Invalid value in {col}; "
            "expected 0-1."
        )


# ============================================================
# UNBLIND
# ============================================================

merged = review.merge(
    key[
        [
            "experiment_id",
            "condition_A",
            "condition_B",
        ]
    ],
    on="experiment_id",
    how="left",
    validate="one_to_one"
)


valid_conditions = {
    "RAG_ON",
    "RAG_OFF"
}


for col in [
    "condition_A",
    "condition_B",
]:

    invalid = set(
        merged[col].dropna()
    ) - valid_conditions

    if invalid:

        raise RuntimeError(
            f"Unexpected conditions in {col}: "
            f"{invalid}"
        )


# ============================================================
# MAP BLINDED SCORES TO CONDITIONS
# ============================================================

def select_condition_value(
    row,
    condition,
    a_col,
    b_col
):

    if row["condition_A"] == condition:
        return row[a_col]

    if row["condition_B"] == condition:
        return row[b_col]

    raise RuntimeError(
        "Condition mapping error."
    )


merged[
    "RAG_ON_factuality"
] = merged.apply(
    lambda row:
        select_condition_value(
            row,
            "RAG_ON",
            "A_factuality_0_2",
            "B_factuality_0_2"
        ),
    axis=1
)


merged[
    "RAG_ON_grounding"
] = merged.apply(
    lambda row:
        select_condition_value(
            row,
            "RAG_ON",
            "A_grounding_0_2",
            "B_grounding_0_2"
        ),
    axis=1
)


merged[
    "RAG_ON_clarity"
] = merged.apply(
    lambda row:
        select_condition_value(
            row,
            "RAG_ON",
            "A_clarity_0_1",
            "B_clarity_0_1"
        ),
    axis=1
)


merged[
    "RAG_OFF_factuality"
] = merged.apply(
    lambda row:
        select_condition_value(
            row,
            "RAG_OFF",
            "A_factuality_0_2",
            "B_factuality_0_2"
        ),
    axis=1
)


merged[
    "RAG_OFF_grounding"
] = merged.apply(
    lambda row:
        select_condition_value(
            row,
            "RAG_OFF",
            "A_grounding_0_2",
            "B_grounding_0_2"
        ),
    axis=1
)


merged[
    "RAG_OFF_clarity"
] = merged.apply(
    lambda row:
        select_condition_value(
            row,
            "RAG_OFF",
            "A_clarity_0_1",
            "B_clarity_0_1"
        ),
    axis=1
)


# ============================================================
# TOTAL SCORES
# ============================================================

merged[
    "RAG_ON_total"
] = (
    merged["RAG_ON_factuality"]
    + merged["RAG_ON_grounding"]
    + merged["RAG_ON_clarity"]
)


merged[
    "RAG_OFF_total"
] = (
    merged["RAG_OFF_factuality"]
    + merged["RAG_OFF_grounding"]
    + merged["RAG_OFF_clarity"]
)


# ============================================================
# PAIRED DIFFERENCES
# ============================================================

merged[
    "total_difference_ON_minus_OFF"
] = (
    merged["RAG_ON_total"]
    - merged["RAG_OFF_total"]
)


merged[
    "factuality_difference_ON_minus_OFF"
] = (
    merged["RAG_ON_factuality"]
    - merged["RAG_OFF_factuality"]
)


merged[
    "grounding_difference_ON_minus_OFF"
] = (
    merged["RAG_ON_grounding"]
    - merged["RAG_OFF_grounding"]
)


merged[
    "clarity_difference_ON_minus_OFF"
] = (
    merged["RAG_ON_clarity"]
    - merged["RAG_OFF_clarity"]
)


# ============================================================
# WIN / TIE / LOSS COUNTS
# ============================================================

total_diffs = merged[
    "total_difference_ON_minus_OFF"
].to_numpy()


on_wins = int(
    np.sum(total_diffs > 0)
)

ties = int(
    np.sum(total_diffs == 0)
)

off_wins = int(
    np.sum(total_diffs < 0)
)


# ============================================================
# WILCOXON SIGNED-RANK TEST
# ============================================================

def run_wilcoxon(
    on_values,
    off_values
):

    differences = (
        np.asarray(on_values)
        - np.asarray(off_values)
    )

    nonzero = (
        differences != 0
    )

    if not np.any(nonzero):

        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_nonzero_pairs": 0
        }


    result = wilcoxon(
        np.asarray(on_values)[nonzero],
        np.asarray(off_values)[nonzero],
        alternative="two-sided",
        method="auto"
    )


    return {
        "statistic": float(
            result.statistic
        ),
        "p_value": float(
            result.pvalue
        ),
        "n_nonzero_pairs": int(
            np.sum(nonzero)
        )
    }


total_test = run_wilcoxon(
    merged["RAG_ON_total"],
    merged["RAG_OFF_total"]
)


factuality_test = run_wilcoxon(
    merged["RAG_ON_factuality"],
    merged["RAG_OFF_factuality"]
)


grounding_test = run_wilcoxon(
    merged["RAG_ON_grounding"],
    merged["RAG_OFF_grounding"]
)


clarity_test = run_wilcoxon(
    merged["RAG_ON_clarity"],
    merged["RAG_OFF_clarity"]
)


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

total_ci = paired_bootstrap_mean_difference(
    merged[
        "total_difference_ON_minus_OFF"
    ],
    seed=BOOTSTRAP_SEED,
    n_boot=BOOTSTRAP_RESAMPLES
)


factuality_ci = paired_bootstrap_mean_difference(
    merged[
        "factuality_difference_ON_minus_OFF"
    ],
    seed=BOOTSTRAP_SEED + 1,
    n_boot=BOOTSTRAP_RESAMPLES
)


grounding_ci = paired_bootstrap_mean_difference(
    merged[
        "grounding_difference_ON_minus_OFF"
    ],
    seed=BOOTSTRAP_SEED + 2,
    n_boot=BOOTSTRAP_RESAMPLES
)


clarity_ci = paired_bootstrap_mean_difference(
    merged[
        "clarity_difference_ON_minus_OFF"
    ],
    seed=BOOTSTRAP_SEED + 3,
    n_boot=BOOTSTRAP_RESAMPLES
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("==============================")
print("SCORE SUMMARY")
print("==============================")


on_summary = summarize(
    merged["RAG_ON_total"]
)

off_summary = summarize(
    merged["RAG_OFF_total"]
)


print()
print("RAG ON total score / 5")
print(
    "Mean:",
    round(on_summary["mean"], 4)
)
print(
    "Median:",
    round(on_summary["median"], 4)
)
print(
    "SD:",
    round(on_summary["std"], 4)
)


print()
print("RAG OFF total score / 5")
print(
    "Mean:",
    round(off_summary["mean"], 4)
)
print(
    "Median:",
    round(off_summary["median"], 4)
)
print(
    "SD:",
    round(off_summary["std"], 4)
)


print()
print(
    "Mean paired difference "
    "(RAG ON - RAG OFF):",
    round(
        np.mean(total_diffs),
        4
    )
)


print(
    "95% bootstrap CI:",
    (
        round(total_ci[0], 4),
        round(total_ci[1], 4)
    )
)


print()
print("==============================")
print("PAIRED COMPARISON")
print("==============================")


print(
    "RAG ON > RAG OFF:",
    on_wins
)

print(
    "Ties:",
    ties
)

print(
    "RAG ON < RAG OFF:",
    off_wins
)


print()
print("Wilcoxon signed-rank test")
print(
    "Statistic:",
    total_test["statistic"]
)

print(
    "p-value:",
    total_test["p_value"]
)

print(
    "Non-zero pairs:",
    total_test["n_nonzero_pairs"]
)


print()
print("==============================")
print("COMPONENT-LEVEL RESULTS")
print("==============================")


component_results = {

    "factuality": {
        "ON_mean":
            float(
                merged["RAG_ON_factuality"].mean()
            ),
        "OFF_mean":
            float(
                merged["RAG_OFF_factuality"].mean()
            ),
        "mean_difference":
            float(
                merged[
                    "factuality_difference_ON_minus_OFF"
                ].mean()
            ),
        "95_bootstrap_CI":
            list(factuality_ci),
        "wilcoxon_statistic":
            factuality_test["statistic"],
        "p_value":
            factuality_test["p_value"],
    },

    "grounding": {
        "ON_mean":
            float(
                merged["RAG_ON_grounding"].mean()
            ),
        "OFF_mean":
            float(
                merged["RAG_OFF_grounding"].mean()
            ),
        "mean_difference":
            float(
                merged[
                    "grounding_difference_ON_minus_OFF"
                ].mean()
            ),
        "95_bootstrap_CI":
            list(grounding_ci),
        "wilcoxon_statistic":
            grounding_test["statistic"],
        "p_value":
            grounding_test["p_value"],
    },

    "clarity": {
        "ON_mean":
            float(
                merged["RAG_ON_clarity"].mean()
            ),
        "OFF_mean":
            float(
                merged["RAG_OFF_clarity"].mean()
            ),
        "mean_difference":
            float(
                merged[
                    "clarity_difference_ON_minus_OFF"
                ].mean()
            ),
        "95_bootstrap_CI":
            list(clarity_ci),
        "wilcoxon_statistic":
            clarity_test["statistic"],
        "p_value":
            clarity_test["p_value"],
    }
}


for name, result in component_results.items():

    print()
    print(name.upper())

    print(
        "RAG ON mean:",
        round(result["ON_mean"], 4)
    )

    print(
        "RAG OFF mean:",
        round(result["OFF_mean"], 4)
    )

    print(
        "Difference:",
        round(result["mean_difference"], 4)
    )

    print(
        "95% bootstrap CI:",
        [
            round(x, 4)
            for x in result["95_bootstrap_CI"]
        ]
    )

    print(
        "Wilcoxon statistic:",
        result["wilcoxon_statistic"]
    )

    print(
        "p-value:",
        result["p_value"]
    )


# ============================================================
# SAVE ROW-LEVEL RESULTS
# ============================================================

output_columns = [

    "experiment_id",
    "chapter",
    "question",
    "reference_answer",

    "condition_A",
    "condition_B",

    "A_factuality_0_2",
    "A_grounding_0_2",
    "A_clarity_0_1",

    "B_factuality_0_2",
    "B_grounding_0_2",
    "B_clarity_0_1",

    "RAG_ON_factuality",
    "RAG_ON_grounding",
    "RAG_ON_clarity",
    "RAG_ON_total",

    "RAG_OFF_factuality",
    "RAG_OFF_grounding",
    "RAG_OFF_clarity",
    "RAG_OFF_total",

    "total_difference_ON_minus_OFF",
    "factuality_difference_ON_minus_OFF",
    "grounding_difference_ON_minus_OFF",
    "clarity_difference_ON_minus_OFF",
]


merged[
    output_columns
].to_csv(
    OUTPUT_RESULTS,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {

    "n_questions":
        EXPECTED_N,

    "sampling_seed":
        42,

    "generation_model":
        "tiiuae/falcon-rw-1b",

    "embedding_model":
        "sentence-transformers/all-MiniLM-L6-v2",

    "retrieval_k":
        3,

    "design":
        "paired RAG ON vs RAG OFF",

    "blinded_human_review":
        True,

    "primary_score":
        "factuality + grounding + clarity, maximum 5",

    "rag_on_total":
        on_summary,

    "rag_off_total":
        off_summary,

    "paired_difference": {
        "mean":
            float(
                np.mean(total_diffs)
            ),
        "median":
            float(
                np.median(total_diffs)
            ),
        "95_percent_bootstrap_CI":
            list(total_ci),
    },

    "pairwise_counts": {
        "rag_on_higher":
            on_wins,
        "ties":
            ties,
        "rag_on_lower":
            off_wins,
    },

    "primary_wilcoxon": total_test,

    "component_results":
        component_results,

    "bootstrap": {
        "seed":
            BOOTSTRAP_SEED,
        "resamples":
            BOOTSTRAP_RESAMPLES,
        "ci":
            "percentile 95% CI"
    }
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
# DONE
# ============================================================

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
print(
    "RAG ON/OFF analysis complete."
)