#!/usr/bin/env python3
"""
Generate three supplementary tables for the AF3 cocaine-versus-morphine
aptamer revision.

Required input files
--------------------
- per_aptamer_per_condition.csv
- per_prediction_values.csv
- run_metadata.json (optional but strongly recommended)

Outputs
-------
S2_table_global_ptm.csv / .tex
    Global pTM summaries across the eight modeled conditions. The table
    reports absolute cocaine and morphine pTM values and the number of
    aptamers for which cocaine pTM exceeds morphine pTM, for both the
    AF3 top-ranked prediction and the seed-level average.

S3_table_top5_by_iptm.csv / .tex
    The five aptamers with the highest cross-condition seed-level average
    cocaine aptamer-ligand chain-pair iPTM, together with corresponding
    global pTM values.

S4_table_top5_by_ptm.csv / .tex
    The five aptamers with the highest cross-condition seed-level average
    cocaine global pTM, together with corresponding aptamer-ligand
    chain-pair iPTM values.

The S2--S4 table .tex files are standalone pdfLaTeX documents that can
be compiled in Overleaf. supporting_table_captions.tex is a LaTeX fragment
for insertion into the manuscript and is not a standalone document. The CSV
files retain separate numeric mean and SD columns for machine-readable reuse.

Definitions
-----------
AF3 top-ranked prediction
    The parent-level AF3 output selected by AF3's ranking procedure.

Seed-level average
    For each aptamer, condition, and ligand, the five samples within each
    seed are averaged first, and the resulting nine seed means are then
    averaged. Because every seed contains five samples, this mean equals
    the pooled mean across all 45 predictions.

Cross-condition ranking
    Aptamers are ranked by the arithmetic mean of their eight
    condition-specific seed-level averages for the cocaine complex.
    S3 ranks by chain_pair_iptm[0][1]; S4 ranks by global ptm.

Example
-------
python3 make_af3_supporting_tables.py \
    --data-dir /mnt/ssd/af3_revision_data \
    --outdir /mnt/ssd/af3_supporting_tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONDITION_ORDER = [
    "buffer_optimized",
    "buffer_proportional",
    "Mg0",
    "Mg2",
    "Mg4",
    "Mg6",
    "Mg8",
    "Mg10",
]

CONDITION_LABELS = {
    "buffer_optimized": "Buffer (optimized)",
    "buffer_proportional": "Buffer (proportional)",
    "Mg0": "Mg0",
    "Mg2": "Mg2",
    "Mg4": "Mg4",
    "Mg6": "Mg6",
    "Mg8": "Mg8",
    "Mg10": "Mg10",
}

EXPECTED_APTAMERS = 71
EXPECTED_CONDITIONS = 8
EXPECTED_TARGETS = {"cocaine", "morphine"}
EXPECTED_SEEDS = 9
EXPECTED_SAMPLES_PER_SEED = 5
TOP_N = 5
EXPECTED_PREDICTIONS_PER_JOB = EXPECTED_SEEDS * EXPECTED_SAMPLES_PER_SEED
EXPECTED_TOTAL_PREDICTIONS = (
    EXPECTED_APTAMERS
    * EXPECTED_CONDITIONS
    * len(EXPECTED_TARGETS)
    * EXPECTED_PREDICTIONS_PER_JOB
)

# The extraction script writes aggregate values rounded to four decimals.
# A discrepancy up to half of one unit in the fourth decimal is therefore
# expected when those values are compared with means recomputed from the
# prediction-level file.
SUMMARY_ROUNDING_TOLERANCE = 5.1e-5
NUMERICAL_TIE_TOLERANCE = 1e-12


APTAMER_REQUIRED_COLUMNS = {
    "aptamer_id",
    "condition",
    "coc_iptm_top",
    "mor_iptm_top",
    "coc_ptm_top",
    "mor_ptm_top",
    "coc_iptm_pred_mean",
    "mor_iptm_pred_mean",
    "coc_ptm_pred_mean",
    "mor_ptm_pred_mean",
}

PREDICTION_REQUIRED_COLUMNS = {
    "aptamer_id",
    "condition",
    "target",
    "seed",
    "sample",
    "ptm",
    "iptm",
}


LATEX_SPECIALS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AF3 supplementary tables S2-S4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the AF3 CSV files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("af3_supporting_tables"),
        help="Directory in which tables are written.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places shown in the LaTeX tables.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def latex_escape(value: object) -> str:
    text = str(value)
    return "".join(LATEX_SPECIALS.get(char, char) for char in text)


def require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    filename: str,
) -> None:
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")


def numeric_column(
    dataframe: pd.DataFrame,
    column: str,
    filename: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.Series:
    values = pd.to_numeric(dataframe[column], errors="coerce")
    if values.isna().any():
        bad_rows = dataframe.loc[values.isna(), [column]].head()
        raise ValueError(
            f"{filename}: missing or non-numeric values in {column}:\n"
            f"{bad_rows.to_string(index=False)}"
        )
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{filename}: non-finite values in {column}")
    if minimum is not None and (array < minimum).any():
        raise ValueError(
            f"{filename}: {column} contains values below {minimum}"
        )
    if maximum is not None and (array > maximum).any():
        raise ValueError(
            f"{filename}: {column} contains values above {maximum}"
        )
    return values.astype(float)


def integer_column(
    dataframe: pd.DataFrame,
    column: str,
    filename: str,
    minimum: int | None = None,
) -> pd.Series:
    """Validate an integer-valued identifier column without truncation."""
    values = pd.to_numeric(dataframe[column], errors="coerce")
    if values.isna().any():
        bad_rows = dataframe.loc[values.isna(), [column]].head()
        raise ValueError(
            f"{filename}: missing or non-numeric values in {column}:\n"
            f"{bad_rows.to_string(index=False)}"
        )
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{filename}: non-finite values in {column}")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        bad = dataframe.loc[~np.isclose(array, rounded, rtol=0.0, atol=0.0), [column]].head()
        raise ValueError(
            f"{filename}: {column} must contain integer values:\n"
            f"{bad.to_string(index=False)}"
        )
    integers = rounded.astype(np.int64)
    if minimum is not None and (integers < minimum).any():
        raise ValueError(
            f"{filename}: {column} contains values below {minimum}"
        )
    return pd.Series(integers, index=dataframe.index, name=column)


def validate_metadata(metadata_path: Path) -> dict[str, object] | None:
    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("n_predictions") != EXPECTED_TOTAL_PREDICTIONS:
        raise ValueError(
            "run_metadata.json reports an unexpected prediction count: "
            f"{metadata.get('n_predictions')}"
        )
    if metadata.get("complete") is not True:
        raise ValueError("run_metadata.json does not report complete=true")
    if metadata.get("warnings"):
        raise ValueError(
            f"run_metadata.json contains warnings: {metadata['warnings']}"
        )
    if metadata.get("conditions") != CONDITION_ORDER:
        raise ValueError(
            "run_metadata.json condition order/set does not match the "
            f"expected conditions: {metadata.get('conditions')}"
        )
    if metadata.get("expected_per_target") != EXPECTED_APTAMERS:
        raise ValueError(
            "run_metadata.json does not report 71 expected aptamers per target"
        )
    if metadata.get("strict") is not True:
        raise ValueError("run_metadata.json does not report strict=true")
    if metadata.get("chain_check") is not True:
        raise ValueError("run_metadata.json does not report chain_check=true")

    aptamers_per_condition = metadata.get("aptamers_per_condition", {})
    expected_counts = {condition: EXPECTED_APTAMERS for condition in CONDITION_ORDER}
    if aptamers_per_condition != expected_counts:
        raise ValueError(
            "run_metadata.json has unexpected aptamer counts by condition: "
            f"{aptamers_per_condition}"
        )
    return metadata


def validate_and_prepare(
    per_aptamer: pd.DataFrame,
    per_prediction: pd.DataFrame,
    metadata_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    require_columns(
        per_aptamer,
        APTAMER_REQUIRED_COLUMNS,
        "per_aptamer_per_condition.csv",
    )
    require_columns(
        per_prediction,
        PREDICTION_REQUIRED_COLUMNS,
        "per_prediction_values.csv",
    )
    validate_metadata(metadata_path)

    per_aptamer = per_aptamer.copy()
    per_prediction = per_prediction.copy()

    for dataframe, filename, columns in [
        (
            per_aptamer,
            "per_aptamer_per_condition.csv",
            ["aptamer_id", "condition"],
        ),
        (
            per_prediction,
            "per_prediction_values.csv",
            ["aptamer_id", "condition", "target"],
        ),
    ]:
        for column in columns:
            if dataframe[column].isna().any():
                raise ValueError(f"{filename}: missing values in {column}")

    # Normalize identifier fields without changing their scientific content.
    for dataframe in (per_aptamer, per_prediction):
        dataframe["aptamer_id"] = dataframe["aptamer_id"].astype(str).str.strip()
        dataframe["condition"] = dataframe["condition"].astype(str).str.strip()
    per_prediction["target"] = (
        per_prediction["target"].astype(str).str.strip().str.lower()
    )

    if per_aptamer["aptamer_id"].eq("").any():
        raise ValueError("Empty aptamer IDs in per_aptamer_per_condition.csv")
    if per_prediction["aptamer_id"].eq("").any():
        raise ValueError("Empty aptamer IDs in per_prediction_values.csv")

    if len(per_aptamer) != EXPECTED_APTAMERS * EXPECTED_CONDITIONS:
        raise ValueError(
            "per_aptamer_per_condition.csv must contain exactly "
            f"{EXPECTED_APTAMERS * EXPECTED_CONDITIONS} rows; "
            f"found {len(per_aptamer)}"
        )
    if per_aptamer["aptamer_id"].nunique() != EXPECTED_APTAMERS:
        raise ValueError("Expected exactly 71 unique aptamers in summary data")
    if set(per_aptamer["condition"]) != set(CONDITION_ORDER):
        raise ValueError(
            "Unexpected conditions in per_aptamer_per_condition.csv: "
            f"{sorted(set(per_aptamer['condition']))}"
        )
    if per_aptamer.duplicated(["aptamer_id", "condition"]).any():
        raise ValueError("Duplicate aptamer-condition rows in summary data")

    summary_condition_counts = per_aptamer.groupby("aptamer_id")[
        "condition"
    ].nunique()
    if not (summary_condition_counts == EXPECTED_CONDITIONS).all():
        raise ValueError(
            "At least one aptamer does not contain all eight summary conditions"
        )

    if len(per_prediction) != EXPECTED_TOTAL_PREDICTIONS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_PREDICTIONS:,} prediction rows; "
            f"found {len(per_prediction):,}"
        )
    if per_prediction["aptamer_id"].nunique() != EXPECTED_APTAMERS:
        raise ValueError("Expected exactly 71 unique aptamers in prediction data")
    if set(per_prediction["condition"]) != set(CONDITION_ORDER):
        raise ValueError(
            "Unexpected conditions in per_prediction_values.csv: "
            f"{sorted(set(per_prediction['condition']))}"
        )
    if set(per_prediction["target"]) != EXPECTED_TARGETS:
        raise ValueError(
            "Prediction targets must be exactly cocaine and morphine"
        )
    if per_prediction.duplicated(
        ["aptamer_id", "condition", "target", "seed", "sample"]
    ).any():
        raise ValueError("Duplicate prediction identifiers detected")

    summary_aptamers = set(per_aptamer["aptamer_id"])
    prediction_aptamers = set(per_prediction["aptamer_id"])
    if summary_aptamers != prediction_aptamers:
        raise ValueError(
            "Aptamer sets differ between the summary and prediction files"
        )

    for column in [
        "coc_iptm_top",
        "mor_iptm_top",
        "coc_ptm_top",
        "mor_ptm_top",
        "coc_iptm_pred_mean",
        "mor_iptm_pred_mean",
        "coc_ptm_pred_mean",
        "mor_ptm_pred_mean",
    ]:
        per_aptamer[column] = numeric_column(
            per_aptamer,
            column,
            "per_aptamer_per_condition.csv",
            minimum=0.0,
            maximum=1.0,
        )

    per_prediction["iptm"] = numeric_column(
        per_prediction,
        "iptm",
        "per_prediction_values.csv",
        minimum=0.0,
        maximum=1.0,
    )
    per_prediction["ptm"] = numeric_column(
        per_prediction,
        "ptm",
        "per_prediction_values.csv",
        minimum=0.0,
        maximum=1.0,
    )
    per_prediction["seed"] = integer_column(
        per_prediction,
        "seed",
        "per_prediction_values.csv",
        minimum=0,
    )
    per_prediction["sample"] = integer_column(
        per_prediction,
        "sample",
        "per_prediction_values.csv",
        minimum=0,
    )

    job_grid = (
        per_prediction.groupby(
            ["aptamer_id", "condition", "target"],
            observed=True,
        )
        .agg(
            n_predictions=("sample", "size"),
            n_seeds=("seed", "nunique"),
            n_samples=("sample", "nunique"),
        )
        .reset_index()
    )
    expected_jobs = EXPECTED_APTAMERS * EXPECTED_CONDITIONS * len(EXPECTED_TARGETS)
    if len(job_grid) != expected_jobs:
        raise ValueError(
            f"Expected {expected_jobs} aptamer-condition-target jobs; "
            f"found {len(job_grid)}"
        )
    bad_jobs = job_grid.loc[
        (job_grid["n_predictions"] != EXPECTED_PREDICTIONS_PER_JOB)
        | (job_grid["n_seeds"] != EXPECTED_SEEDS)
        | (job_grid["n_samples"] != EXPECTED_SAMPLES_PER_SEED)
    ]
    if not bad_jobs.empty:
        raise ValueError(
            "Incomplete 9 x 5 prediction grids detected:\n"
            f"{bad_jobs.head().to_string(index=False)}"
        )

    # Infer the expected seed set from the data, then require the same exact
    # set in every job. The exact seed integers are study-specific.
    first_job_key = tuple(job_grid.iloc[0][["aptamer_id", "condition", "target"]])
    first_job = per_prediction.loc[
        (per_prediction["aptamer_id"] == first_job_key[0])
        & (per_prediction["condition"] == first_job_key[1])
        & (per_prediction["target"] == first_job_key[2])
    ]
    expected_seed_ids = sorted(first_job["seed"].unique().tolist())
    expected_sample_ids = sorted(first_job["sample"].unique().tolist())
    if len(expected_seed_ids) != EXPECTED_SEEDS:
        raise ValueError("Could not infer exactly nine seed IDs")
    if expected_sample_ids != list(range(EXPECTED_SAMPLES_PER_SEED)):
        raise ValueError(
            "Expected sample IDs [0, 1, 2, 3, 4], found "
            f"{expected_sample_ids}"
        )

    for key, group in per_prediction.groupby(
        ["aptamer_id", "condition", "target"], observed=True
    ):
        if sorted(group["seed"].unique().tolist()) != expected_seed_ids:
            raise ValueError(f"Job {key} uses a different seed set")
        if sorted(group["sample"].unique().tolist()) != expected_sample_ids:
            raise ValueError(f"Job {key} uses a different sample set")
        per_seed_counts = group.groupby("seed")["sample"].nunique()
        if not (per_seed_counts == EXPECTED_SAMPLES_PER_SEED).all():
            raise ValueError(f"Job {key} has an incomplete seed-sample grid")

    # Recompute the seed-level averages directly from the prediction-level
    # records: five samples per seed, then nine seed means per job.
    seed_means = (
        per_prediction.groupby(
            ["aptamer_id", "condition", "target", "seed"],
            observed=True,
            as_index=False,
        )[["iptm", "ptm"]]
        .mean()
    )
    job_means_long = (
        seed_means.groupby(
            ["aptamer_id", "condition", "target"],
            observed=True,
            as_index=False,
        )[["iptm", "ptm"]]
        .mean()
        .rename(
            columns={
                "iptm": "seed_iptm",
                "ptm": "seed_ptm",
            }
        )
    )

    job_means = job_means_long.pivot(
        index=["aptamer_id", "condition"],
        columns="target",
        values=["seed_iptm", "seed_ptm"],
    )
    job_means.columns = [f"{metric}_{target}" for metric, target in job_means.columns]
    job_means = job_means.reset_index()

    required_job_columns = {
        "seed_iptm_cocaine",
        "seed_iptm_morphine",
        "seed_ptm_cocaine",
        "seed_ptm_morphine",
    }
    if required_job_columns - set(job_means.columns):
        raise ValueError("Could not construct complete cocaine/morphine job means")

    merged = per_aptamer.merge(
        job_means,
        on=["aptamer_id", "condition"],
        how="left",
        validate="one_to_one",
    )
    if merged[list(required_job_columns)].isna().any().any():
        raise ValueError("Missing recomputed job means after merging")

    comparisons = {
        "coc_iptm_pred_mean": "seed_iptm_cocaine",
        "mor_iptm_pred_mean": "seed_iptm_morphine",
        "coc_ptm_pred_mean": "seed_ptm_cocaine",
        "mor_ptm_pred_mean": "seed_ptm_morphine",
    }
    for stored_column, recomputed_column in comparisons.items():
        differences = (
            merged[stored_column] - merged[recomputed_column]
        ).abs()
        if differences.max() > SUMMARY_ROUNDING_TOLERANCE:
            worst_index = differences.idxmax()
            row = merged.loc[
                worst_index,
                [
                    "aptamer_id",
                    "condition",
                    stored_column,
                    recomputed_column,
                ],
            ]
            raise ValueError(
                "Stored and recomputed seed-level means disagree beyond "
                f"rounding tolerance:\n{row.to_string()}"
            )

    merged["condition"] = pd.Categorical(
        merged["condition"], categories=CONDITION_ORDER, ordered=True
    )
    job_means_long["condition"] = pd.Categorical(
        job_means_long["condition"], categories=CONDITION_ORDER, ordered=True
    )

    return merged, per_prediction, job_means_long, expected_seed_ids, expected_sample_ids


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty set of values")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def classify_pair(
    cocaine: pd.Series,
    morphine: pd.Series,
) -> tuple[int, int, int]:
    difference = cocaine.to_numpy(dtype=float) - morphine.to_numpy(dtype=float)
    ties = np.isclose(
        difference,
        0.0,
        rtol=0.0,
        atol=NUMERICAL_TIE_TOLERANCE,
    )
    cocaine_wins = difference > NUMERICAL_TIE_TOLERANCE
    morphine_wins = difference < -NUMERICAL_TIE_TOLERANCE
    return (
        int(cocaine_wins.sum()),
        int(ties.sum()),
        int(morphine_wins.sum()),
    )


def build_s2_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for condition in CONDITION_ORDER:
        subset = merged.loc[merged["condition"].astype(str) == condition].copy()
        if len(subset) != EXPECTED_APTAMERS:
            raise ValueError(
                f"Condition {condition} has {len(subset)} rows, expected 71"
            )

        top_coc_mean, top_coc_sd = mean_sd(subset["coc_ptm_top"])
        top_mor_mean, top_mor_sd = mean_sd(subset["mor_ptm_top"])
        top_delta = subset["coc_ptm_top"] - subset["mor_ptm_top"]
        top_delta_mean, top_delta_sd = mean_sd(top_delta)
        top_wins, top_ties, top_losses = classify_pair(
            subset["coc_ptm_top"], subset["mor_ptm_top"]
        )

        seed_coc_mean, seed_coc_sd = mean_sd(subset["seed_ptm_cocaine"])
        seed_mor_mean, seed_mor_sd = mean_sd(subset["seed_ptm_morphine"])
        seed_delta = subset["seed_ptm_cocaine"] - subset["seed_ptm_morphine"]
        seed_delta_mean, seed_delta_sd = mean_sd(seed_delta)
        seed_wins, seed_ties, seed_losses = classify_pair(
            subset["seed_ptm_cocaine"], subset["seed_ptm_morphine"]
        )

        rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "n_aptamers": EXPECTED_APTAMERS,
                "top_cocaine_ptm_mean": top_coc_mean,
                "top_cocaine_ptm_sd": top_coc_sd,
                "top_morphine_ptm_mean": top_mor_mean,
                "top_morphine_ptm_sd": top_mor_sd,
                "top_delta_ptm_mean": top_delta_mean,
                "top_delta_ptm_sd": top_delta_sd,
                "top_cocaine_wins": top_wins,
                "top_ties": top_ties,
                "top_morphine_wins": top_losses,
                "top_cocaine_win_percent": 100.0 * top_wins / EXPECTED_APTAMERS,
                "seed_cocaine_ptm_mean": seed_coc_mean,
                "seed_cocaine_ptm_sd": seed_coc_sd,
                "seed_morphine_ptm_mean": seed_mor_mean,
                "seed_morphine_ptm_sd": seed_mor_sd,
                "seed_delta_ptm_mean": seed_delta_mean,
                "seed_delta_ptm_sd": seed_delta_sd,
                "seed_cocaine_wins": seed_wins,
                "seed_ties": seed_ties,
                "seed_morphine_wins": seed_losses,
                "seed_cocaine_win_percent": 100.0 * seed_wins / EXPECTED_APTAMERS,
            }
        )

    return pd.DataFrame(rows)


def cross_condition_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for aptamer_id, subset in merged.groupby("aptamer_id", observed=True):
        if len(subset) != EXPECTED_CONDITIONS:
            raise ValueError(
                f"{aptamer_id} has {len(subset)} conditions, expected 8"
            )
        if set(subset["condition"].astype(str)) != set(CONDITION_ORDER):
            raise ValueError(f"{aptamer_id} does not contain the expected conditions")

        record: dict[str, object] = {"aptamer_id": aptamer_id}
        metric_columns = {
            "top_cocaine_iptm": "coc_iptm_top",
            "top_morphine_iptm": "mor_iptm_top",
            "seed_cocaine_iptm": "seed_iptm_cocaine",
            "seed_morphine_iptm": "seed_iptm_morphine",
            "top_cocaine_ptm": "coc_ptm_top",
            "top_morphine_ptm": "mor_ptm_top",
            "seed_cocaine_ptm": "seed_ptm_cocaine",
            "seed_morphine_ptm": "seed_ptm_morphine",
        }
        for prefix, column in metric_columns.items():
            metric_mean, metric_sd = mean_sd(subset[column])
            record[f"{prefix}_mean"] = metric_mean
            record[f"{prefix}_sd"] = metric_sd

        if record["seed_morphine_iptm_mean"] <= 0:
            raise ValueError(
                f"{aptamer_id} has non-positive morphine iPTM denominator"
            )
        record["seed_iptm_cocaine_over_morphine"] = (
            record["seed_cocaine_iptm_mean"]
            / record["seed_morphine_iptm_mean"]
        )
        record["seed_ptm_cocaine_minus_morphine"] = (
            record["seed_cocaine_ptm_mean"]
            - record["seed_morphine_ptm_mean"]
        )
        rows.append(record)

    return pd.DataFrame(rows)


def rank_top_aptamers(
    summary: pd.DataFrame,
    ranking_column: str,
) -> pd.DataFrame:
    """Rank by one prespecified metric; resolve exact ties by aptamer ID."""
    if ranking_column not in summary.columns:
        raise ValueError(f"Unknown ranking column: {ranking_column}")

    ranked = summary.copy()
    ranked["_natural_key"] = ranked["aptamer_id"].map(natural_key)

    # First establish the neutral natural-ID tie order, then use a stable
    # descending sort on the sole scientific ranking metric. This avoids
    # introducing an undeclared secondary biological criterion.
    ranked = ranked.sort_values(
        "_natural_key", ascending=True, kind="mergesort"
    )
    ranked = ranked.sort_values(
        ranking_column, ascending=False, kind="mergesort"
    )

    ranked = ranked.head(TOP_N).drop(columns="_natural_key").copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def validate_output_tables(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    s4: pd.DataFrame,
) -> None:
    if s2["condition"].tolist() != CONDITION_ORDER:
        raise ValueError("S2 condition order is incorrect")
    for prefix in ("top", "seed"):
        totals = (
            s2[f"{prefix}_cocaine_wins"]
            + s2[f"{prefix}_ties"]
            + s2[f"{prefix}_morphine_wins"]
        )
        if not (totals == EXPECTED_APTAMERS).all():
            raise ValueError(
                f"S2 {prefix}-level win/tie/loss counts do not sum to 71"
            )

    for table, metric, label in [
        (s3, "seed_cocaine_iptm_mean", "S3"),
        (s4, "seed_cocaine_ptm_mean", "S4"),
    ]:
        if len(table) != TOP_N:
            raise ValueError(f"{label} must contain exactly {TOP_N} rows")
        if table["aptamer_id"].duplicated().any():
            raise ValueError(f"{label} contains duplicate aptamer IDs")
        if table["rank"].tolist() != list(range(1, TOP_N + 1)):
            raise ValueError(f"{label} ranks are not consecutive")
        values = table[metric].to_numpy(dtype=float)
        if np.any(values[:-1] < values[1:]):
            raise ValueError(f"{label} is not sorted by descending {metric}")


def fmt_mean_sd(mean: float, sd: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}"


def fmt_count(wins: int, total: int, percent: float, ties: int) -> str:
    tie_word = "tie" if ties == 1 else "ties"
    return f"{wins}/{total} ({percent:.1f}\\%; {ties} {tie_word})"


def latex_document_start(title: str, font_size: str = r"\scriptsize") -> str:
    return rf"""\documentclass[10pt]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[margin=0.55in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{longtable}}
\usepackage{{pdflscape}}
\usepackage{{array}}
\usepackage{{ragged2e}}
\setlength{{\LTpre}}{{0pt}}
\setlength{{\LTpost}}{{0pt}}
\begin{{document}}
\begin{{landscape}}
\begin{{center}}
{{\large\textbf{{{title}}}}}
\end{{center}}
{font_size}
"""


def latex_document_end(note: str) -> str:
    return rf"""
\vspace{{0.7em}}
\noindent\textit{{Note.}} {note}
\end{{landscape}}
\end{{document}}
"""


def write_s2_latex(table: pd.DataFrame, path: Path, decimals: int) -> None:
    title = (
        "S2 Table. Global pTM confidence and cocaine--morphine "
        "comparison across modeled conditions"
    )
    lines = [latex_document_start(title)]
    lines.extend(
        [
            r"\begin{longtable}{@{}lcccccc@{}}",
            r"\toprule",
            r"& \multicolumn{3}{c}{\textbf{AF3 top-ranked prediction}} & "
            r"\multicolumn{3}{c}{\textbf{Seed-level average}} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"\textbf{Condition} & "
            r"\textbf{Cocaine pTM} & \textbf{Morphine pTM} & "
            r"\textbf{Cocaine $>$ morphine} & "
            r"\textbf{Cocaine pTM} & \textbf{Morphine pTM} & "
            r"\textbf{Cocaine $>$ morphine} \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"& \multicolumn{3}{c}{\textbf{AF3 top-ranked prediction}} & "
            r"\multicolumn{3}{c}{\textbf{Seed-level average}} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"\textbf{Condition} & "
            r"\textbf{Cocaine pTM} & \textbf{Morphine pTM} & "
            r"\textbf{Cocaine $>$ morphine} & "
            r"\textbf{Cocaine pTM} & \textbf{Morphine pTM} & "
            r"\textbf{Cocaine $>$ morphine} \\",
            r"\midrule",
            r"\endhead",
        ]
    )

    for row in table.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    latex_escape(row.condition_label),
                    fmt_mean_sd(
                        row.top_cocaine_ptm_mean,
                        row.top_cocaine_ptm_sd,
                        decimals,
                    ),
                    fmt_mean_sd(
                        row.top_morphine_ptm_mean,
                        row.top_morphine_ptm_sd,
                        decimals,
                    ),
                    fmt_count(
                        row.top_cocaine_wins,
                        row.n_aptamers,
                        row.top_cocaine_win_percent,
                        row.top_ties,
                    ),
                    fmt_mean_sd(
                        row.seed_cocaine_ptm_mean,
                        row.seed_cocaine_ptm_sd,
                        decimals,
                    ),
                    fmt_mean_sd(
                        row.seed_morphine_ptm_mean,
                        row.seed_morphine_ptm_sd,
                        decimals,
                    ),
                    fmt_count(
                        row.seed_cocaine_wins,
                        row.n_aptamers,
                        row.seed_cocaine_win_percent,
                        row.seed_ties,
                    ),
                ]
            )
            + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{longtable}"])
    note = (
        "pTM is the global confidence metric for the complete modeled complex. "
        "Values are mean $\\pm$ SD across the 71 aptamers within each condition. "
        "The comparison columns report the number and percentage of aptamers for "
        "which cocaine pTM exceeded morphine pTM; exact numerical ties remain in "
        "the denominator and are reported explicitly. Seed-level averages were "
        "calculated by averaging the five samples within each seed and then "
        "averaging across the nine seeds."
    )
    lines.append(latex_document_end(note))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_s3_latex(table: pd.DataFrame, path: Path, decimals: int) -> None:
    title = (
        "S3 Table. Five aptamers with the highest mean cocaine "
        "aptamer--ligand iPTM across modeled conditions under seed-level "
        "averaging and their corresponding global pTM values"
    )
    lines = [latex_document_start(title, font_size=r"\tiny")]
    lines.extend(
        [
            r"\begin{longtable}{@{}r l cc ccc cc cc@{}}",
            r"\toprule",
            r"& & \multicolumn{2}{c}{\textbf{AF3 top-ranked iPTM}} & "
            r"\multicolumn{3}{c}{\textbf{Seed-level average iPTM}} & "
            r"\multicolumn{2}{c}{\textbf{AF3 top-ranked global pTM}} & "
            r"\multicolumn{2}{c}{\textbf{Seed-level average global pTM}} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}"
            r"\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
            r"\textbf{Rank} & \textbf{Aptamer} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & \textbf{C/M} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & "
            r"\textbf{Cocaine} & \textbf{Morphine} \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"& & \multicolumn{2}{c}{\textbf{AF3 top-ranked iPTM}} & "
            r"\multicolumn{3}{c}{\textbf{Seed-level average iPTM}} & "
            r"\multicolumn{2}{c}{\textbf{AF3 top-ranked global pTM}} & "
            r"\multicolumn{2}{c}{\textbf{Seed-level average global pTM}} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}"
            r"\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
            r"\textbf{Rank} & \textbf{Aptamer} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & \textbf{C/M} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & "
            r"\textbf{Cocaine} & \textbf{Morphine} \\",
            r"\midrule",
            r"\endhead",
        ]
    )

    for row in table.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    str(row.rank),
                    latex_escape(row.aptamer_id),
                    fmt_mean_sd(row.top_cocaine_iptm_mean, row.top_cocaine_iptm_sd, decimals),
                    fmt_mean_sd(row.top_morphine_iptm_mean, row.top_morphine_iptm_sd, decimals),
                    fmt_mean_sd(row.seed_cocaine_iptm_mean, row.seed_cocaine_iptm_sd, decimals),
                    fmt_mean_sd(row.seed_morphine_iptm_mean, row.seed_morphine_iptm_sd, decimals),
                    f"{row.seed_iptm_cocaine_over_morphine:.{decimals}f}",
                    fmt_mean_sd(row.top_cocaine_ptm_mean, row.top_cocaine_ptm_sd, decimals),
                    fmt_mean_sd(row.top_morphine_ptm_mean, row.top_morphine_ptm_sd, decimals),
                    fmt_mean_sd(row.seed_cocaine_ptm_mean, row.seed_cocaine_ptm_sd, decimals),
                    fmt_mean_sd(row.seed_morphine_ptm_mean, row.seed_morphine_ptm_sd, decimals),
                ]
            )
            + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{longtable}"])
    note = (
        "Aptamers were ranked by the arithmetic mean of their eight "
        "condition-specific seed-level average cocaine iPTM values. iPTM is "
        "the aptamer--ligand chain-pair metric "
        "\\texttt{chain\\_pair\\_iptm[0][1]}, whereas pTM is the global "
        "confidence metric for the complete modeled complex. All displayed "
        "mean $\\pm$ SD values summarize variation across the eight modeled "
        "conditions, not variation across seeds. C/M is the ratio of the "
        "cross-condition cocaine and morphine seed-level mean iPTM values; "
        "it is not the arithmetic mean of eight condition-specific ratios."
    )
    lines.append(latex_document_end(note))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_s4_latex(table: pd.DataFrame, path: Path, decimals: int) -> None:
    title = (
        "S4 Table. Five aptamers with the highest mean cocaine global pTM "
        "across modeled conditions under seed-level averaging and their "
        "corresponding aptamer--ligand iPTM values"
    )
    lines = [latex_document_start(title, font_size=r"\tiny")]
    lines.extend(
        [
            r"\begin{longtable}{@{}r l cc ccc cc ccc@{}}",
            r"\toprule",
            r"& & \multicolumn{2}{c}{\textbf{AF3 top-ranked global pTM}} & "
            r"\multicolumn{3}{c}{\textbf{Seed-level average global pTM}} & "
            r"\multicolumn{2}{c}{\textbf{AF3 top-ranked iPTM}} & "
            r"\multicolumn{3}{c}{\textbf{Seed-level average iPTM}} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}"
            r"\cmidrule(lr){8-9}\cmidrule(lr){10-12}",
            r"\textbf{Rank} & \textbf{Aptamer} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & $\boldsymbol{\Delta}$\textbf{pTM} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & \textbf{C/M} \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"& & \multicolumn{2}{c}{\textbf{AF3 top-ranked global pTM}} & "
            r"\multicolumn{3}{c}{\textbf{Seed-level average global pTM}} & "
            r"\multicolumn{2}{c}{\textbf{AF3 top-ranked iPTM}} & "
            r"\multicolumn{3}{c}{\textbf{Seed-level average iPTM}} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}"
            r"\cmidrule(lr){8-9}\cmidrule(lr){10-12}",
            r"\textbf{Rank} & \textbf{Aptamer} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & $\boldsymbol{\Delta}$\textbf{pTM} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & "
            r"\textbf{Cocaine} & \textbf{Morphine} & \textbf{C/M} \\",
            r"\midrule",
            r"\endhead",
        ]
    )

    for row in table.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    str(row.rank),
                    latex_escape(row.aptamer_id),
                    fmt_mean_sd(row.top_cocaine_ptm_mean, row.top_cocaine_ptm_sd, decimals),
                    fmt_mean_sd(row.top_morphine_ptm_mean, row.top_morphine_ptm_sd, decimals),
                    fmt_mean_sd(row.seed_cocaine_ptm_mean, row.seed_cocaine_ptm_sd, decimals),
                    fmt_mean_sd(row.seed_morphine_ptm_mean, row.seed_morphine_ptm_sd, decimals),
                    f"{row.seed_ptm_cocaine_minus_morphine:+.{decimals}f}",
                    fmt_mean_sd(row.top_cocaine_iptm_mean, row.top_cocaine_iptm_sd, decimals),
                    fmt_mean_sd(row.top_morphine_iptm_mean, row.top_morphine_iptm_sd, decimals),
                    fmt_mean_sd(row.seed_cocaine_iptm_mean, row.seed_cocaine_iptm_sd, decimals),
                    fmt_mean_sd(row.seed_morphine_iptm_mean, row.seed_morphine_iptm_sd, decimals),
                    f"{row.seed_iptm_cocaine_over_morphine:.{decimals}f}",
                ]
            )
            + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{longtable}"])
    note = (
        "Aptamers were ranked by the arithmetic mean of their eight "
        "condition-specific seed-level average cocaine global pTM values. "
        "pTM is the global confidence metric for the complete modeled "
        "complex, whereas iPTM is the aptamer--ligand chain-pair metric "
        "\\texttt{chain\\_pair\\_iptm[0][1]}. All displayed mean "
        "$\\pm$ SD values summarize variation across the eight modeled "
        "conditions, not variation across seeds. $\\Delta$pTM is the "
        "cross-condition seed-level cocaine mean minus the corresponding "
        "morphine mean. C/M is the ratio of cross-condition seed-level "
        "cocaine and morphine mean iPTM values."
    )
    lines.append(latex_document_end(note))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_captions(path: Path) -> None:
    text = r"""\noindent\textbf{S2 Table. Global pTM confidence and cocaine--morphine comparison across modeled conditions.}
The table summarizes global pTM for cocaine- and morphine-containing complexes and reports the number of aptamers for which cocaine pTM exceeded morphine pTM under the AF3 top-ranked-prediction analysis and seed-level averaging.

\noindent\textbf{S3 Table. Five aptamers with the highest mean cocaine aptamer--ligand iPTM across modeled conditions under seed-level averaging and their corresponding global pTM values.}
Aptamers were ranked solely by their mean cocaine aptamer--ligand chain-pair iPTM across the eight modeled conditions using seed-level averaging. Corresponding morphine iPTM and global pTM values are also reported. Exact ranking ties, if present, are resolved by natural aptamer identifier order.

\noindent\textbf{S4 Table. Five aptamers with the highest mean cocaine global pTM across modeled conditions under seed-level averaging and their corresponding aptamer--ligand iPTM values.}
Aptamers were ranked solely by their mean cocaine global pTM across the eight modeled conditions using seed-level averaging. Corresponding morphine pTM and aptamer--ligand chain-pair iPTM values are also reported. Exact ranking ties, if present, are resolved by natural aptamer identifier order.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.decimals < 1 or args.decimals > 6:
        raise ValueError("--decimals must be between 1 and 6")

    data_dir = args.data_dir.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = data_dir / "per_aptamer_per_condition.csv"
    prediction_path = data_dir / "per_prediction_values.csv"
    metadata_path = data_dir / "run_metadata.json"

    for required_path in (summary_path, prediction_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required input not found: {required_path}")

    per_aptamer = pd.read_csv(summary_path)
    per_prediction = pd.read_csv(prediction_path)

    merged, _, _, seed_ids, sample_ids = validate_and_prepare(
        per_aptamer,
        per_prediction,
        metadata_path,
    )

    s2 = build_s2_table(merged)
    cross_summary = cross_condition_summary(merged)

    s3 = rank_top_aptamers(
        cross_summary,
        ranking_column="seed_cocaine_iptm_mean",
    )
    s4 = rank_top_aptamers(
        cross_summary,
        ranking_column="seed_cocaine_ptm_mean",
    )

    validate_output_tables(s2, s3, s4)

    s2_csv = outdir / "S2_table_global_ptm.csv"
    s3_csv = outdir / "S3_table_top5_by_iptm.csv"
    s4_csv = outdir / "S4_table_top5_by_ptm.csv"
    s2_tex = outdir / "S2_table_global_ptm.tex"
    s3_tex = outdir / "S3_table_top5_by_iptm.tex"
    s4_tex = outdir / "S4_table_top5_by_ptm.tex"

    s2.to_csv(s2_csv, index=False, float_format="%.8f")
    s3.to_csv(s3_csv, index=False, float_format="%.8f")
    s4.to_csv(s4_csv, index=False, float_format="%.8f")

    write_s2_latex(s2, s2_tex, args.decimals)
    write_s3_latex(s3, s3_tex, args.decimals)
    write_s4_latex(s4, s4_tex, args.decimals)
    write_captions(outdir / "supporting_table_captions.tex")

    script_path = Path(__file__).resolve()
    manifest = {
        "script": script_path.name,
        "script_sha256": sha256_file(script_path),
        "inputs": {
            summary_path.name: sha256_file(summary_path),
            prediction_path.name: sha256_file(prediction_path),
            **(
                {metadata_path.name: sha256_file(metadata_path)}
                if metadata_path.exists()
                else {}
            ),
        },
        "conditions": CONDITION_ORDER,
        "n_aptamers": EXPECTED_APTAMERS,
        "seed_ids": seed_ids,
        "sample_ids": sample_ids,
        "top_n": TOP_N,
        "ranking_tie_break": "Natural aptamer identifier order",
        "s3_ranking_definition": (
            "Descending arithmetic mean of the eight condition-specific "
            "seed-level average cocaine chain-pair iPTM values"
        ),
        "s4_ranking_definition": (
            "Descending arithmetic mean of the eight condition-specific "
            "seed-level average cocaine global pTM values"
        ),
        "outputs": [
            s2_csv.name,
            s2_tex.name,
            s3_csv.name,
            s3_tex.name,
            s4_csv.name,
            s4_tex.name,
            "supporting_table_captions.tex",
        ],
    }
    (outdir / "supporting_tables_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Input validation passed.")
    print(
        f"Validated {EXPECTED_APTAMERS} aptamers x {EXPECTED_CONDITIONS} "
        f"conditions x 2 ligands x {EXPECTED_SEEDS} seeds x "
        f"{EXPECTED_SAMPLES_PER_SEED} samples = "
        f"{EXPECTED_TOTAL_PREDICTIONS:,} prediction records."
    )
    print(f"Seed IDs: {seed_ids}")
    print(f"Sample IDs: {sample_ids}")
    print("\nS3 top aptamers by cross-condition seed-level cocaine iPTM:")
    for row in s3.itertuples(index=False):
        print(
            f"  {row.rank}. {row.aptamer_id}: "
            f"iPTM={row.seed_cocaine_iptm_mean:.6f}, "
            f"pTM={row.seed_cocaine_ptm_mean:.6f}"
        )
    print("\nS4 top aptamers by cross-condition seed-level cocaine global pTM:")
    for row in s4.itertuples(index=False):
        print(
            f"  {row.rank}. {row.aptamer_id}: "
            f"pTM={row.seed_cocaine_ptm_mean:.6f}, "
            f"iPTM={row.seed_cocaine_iptm_mean:.6f}"
        )
    print(f"\nTables written to: {outdir}")
    print("Compile the S2--S4 table .tex files with pdfLaTeX in Overleaf.")
    print("supporting_table_captions.tex is a manuscript fragment, not a standalone document.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
