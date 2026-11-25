"""Dataset loading, validation, and construction of independent animal IDs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

META_COLUMNS = ["MouseID", "Genotype", "Treatment", "Behavior", "class"]
EXPECTED_ROWS = 1080
EXPECTED_PROTEINS = 77


def animal_id(measurement_id: str) -> str:
    """Return the biological animal identifier from a replicate identifier."""
    match = re.fullmatch(r"(.+)_([1-9]|1[0-5])", str(measurement_id))
    if not match:
        raise ValueError(f"Unexpected MouseID format: {measurement_id!r}")
    return match.group(1)


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the original UCI workbook without modifying measurements."""
    frame = pd.read_excel(Path(path), engine="xlrd")
    missing = set(META_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    proteins = [c for c in frame.columns if c not in META_COLUMNS]
    if frame.shape[0] != EXPECTED_ROWS or len(proteins) != EXPECTED_PROTEINS:
        raise ValueError(f"Expected 1080 rows/77 proteins, got {len(frame)}/{len(proteins)}")
    frame = frame.copy()
    frame.insert(1, "AnimalID", frame["MouseID"].map(animal_id))
    counts = frame.groupby("AnimalID").size()
    if frame["AnimalID"].nunique() != 72 or not counts.eq(15).all():
        raise ValueError("Expected 72 animals with exactly 15 repeated measurements each")
    for col in ["Genotype", "Treatment", "Behavior", "class"]:
        if frame.groupby("AnimalID")[col].nunique().max() != 1:
            raise ValueError(f"{col} changes within an animal")
    return frame


def protein_columns(frame: pd.DataFrame) -> list[str]:
    """Return the 77 continuous protein features."""
    return [c for c in frame.columns if c not in META_COLUMNS + ["AnimalID"]]


#: The eight experimental cells as ``genotype-behavior-treatment`` codes used by UCI.
CELLS: tuple[str, ...] = (
    "c-CS-s", "c-SC-s", "c-CS-m", "c-SC-m", "t-CS-s", "t-SC-s", "t-CS-m", "t-SC-m",
)
#: Proteins excluded from multivariate analyses because >= 5 mice lack any value.
SPARSE_MIN_MISSING_MICE = 5


def replicate_index(frame: pd.DataFrame) -> pd.Series:
    """Return the within-animal measurement index (1-15) encoded in ``MouseID``."""
    return frame["MouseID"].str.rsplit("_", n=1).str[1].astype(int)


def dilution_block(frame: pd.DataFrame) -> pd.Series:
    """Infer the technical block (0-4) of each row from its within-animal index.

    The workbook does not label dilution or replicate. Indices 1-3, 4-6, ..., 13-15
    behave as five blocks of three near-identical rows (see the QC analysis), which
    is consistent with three replicates at each of five dilution points. This
    grouping is an inference used only for quality control.
    """
    return (replicate_index(frame) - 1) // 3


def duplicate_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map each protein column that exactly duplicates an earlier one to that column."""
    proteins = protein_columns(frame)
    seen: dict[tuple, str] = {}
    duplicates: dict[str, str] = {}
    for column in proteins:
        key = tuple(frame[column].fillna(-np.inf).to_numpy())
        if key in seen:
            duplicates[column] = seen[key]
        else:
            seen[key] = column
    return duplicates


def log2_animal_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-animal means of log2 protein levels and per-animal labels.

    Values <= 0 cannot be log-transformed; the workbook contains one such value
    (an RRP1 measurement), which is treated as missing. Averaging on the log scale
    gives each animal the geometric mean of its 15 technical measurements, so
    group contrasts are interpretable as log2 fold changes.
    """
    proteins = protein_columns(frame)
    logged = np.log2(frame[proteins].where(frame[proteins] > 0))
    values = logged.groupby(frame["AnimalID"], sort=True).mean()
    labels = frame.groupby("AnimalID", sort=True)[
        ["Genotype", "Treatment", "Behavior", "class"]
    ].first()
    return values, labels


def analysis_panel(frame: pd.DataFrame) -> list[str]:
    """Proteins used in multivariate analyses: non-duplicated and not sparse.

    Excludes exact duplicate columns and proteins with no measurement at all for
    at least ``SPARSE_MIN_MISSING_MICE`` animals.
    """
    values, _ = log2_animal_matrix(frame)
    duplicates = duplicate_columns(frame)
    sparse = values.columns[values.isna().sum() >= SPARSE_MIN_MISSING_MICE]
    return [c for c in values.columns if c not in duplicates and c not in sparse]


def aggregate_animals(frame: pd.DataFrame) -> pd.DataFrame:
    """Average technical repeats per animal while preserving experimental labels."""
    proteins = protein_columns(frame)
    values = frame.groupby("AnimalID", sort=False)[proteins].mean()
    labels = frame.groupby("AnimalID", sort=False)[
        ["Genotype", "Treatment", "Behavior", "class"]
    ].first()
    return values.join(labels).reset_index()
