"""
load_data.py
------------
Creates teiko.db in the project root with a normalized SQLite schema,
then ingests cell-count.csv.

Schema (4 tables, 3NF):
    projects    – one row per project
    subjects    – one row per patient (demographics, treatment, response)
    samples     – one row per biological sample (type, time-point)
    cell_counts – one row per sample with raw cell counts (wide format)

Run:
    python load_data.py
"""

import os
import sqlite3

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "teiko.db")
CSV_PATH = os.path.join(BASE_DIR, "cell-count.csv")

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
DDL_STATEMENTS = [
    "PRAGMA foreign_keys = ON",

    "DROP TABLE IF EXISTS cell_counts",
    "DROP TABLE IF EXISTS samples",
    "DROP TABLE IF EXISTS subjects",
    "DROP TABLE IF EXISTS projects",

    """
    CREATE TABLE projects (
        project_id TEXT PRIMARY KEY
    )
    """,

    """
    CREATE TABLE subjects (
        subject_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(project_id),
        condition  TEXT NOT NULL,
        age        INTEGER,
        sex        TEXT,
        treatment  TEXT,
        response   TEXT
    )
    """,

    """
    CREATE TABLE samples (
        sample_id                 TEXT PRIMARY KEY,
        subject_id                TEXT NOT NULL REFERENCES subjects(subject_id),
        sample_type               TEXT NOT NULL,
        time_from_treatment_start INTEGER NOT NULL
    )
    """,

    """
    CREATE TABLE cell_counts (
        sample_id  TEXT PRIMARY KEY REFERENCES samples(sample_id),
        b_cell     INTEGER NOT NULL,
        cd8_t_cell INTEGER NOT NULL,
        cd4_t_cell INTEGER NOT NULL,
        nk_cell    INTEGER NOT NULL,
        monocyte   INTEGER NOT NULL
    )
    """,
]


def create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for stmt in DDL_STATEMENTS:
        cur.execute(stmt)
    conn.commit()
    print("Schema created (projects, subjects, samples, cell_counts).")


def load_csv(conn: sqlite3.Connection) -> None:
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} rows × {len(df.columns)} columns loaded.")

    # ── projects ──────────────────────────────────────────────────────────────
    projects = (
        df[["project"]]
        .drop_duplicates()
        .rename(columns={"project": "project_id"})
    )
    projects.to_sql("projects", conn, if_exists="append", index=False)
    print(f"  projects   : {len(projects):>6,} rows inserted")

    # ── subjects ──────────────────────────────────────────────────────────────
    subject_cols = ["subject", "project", "condition", "age", "sex", "treatment", "response"]
    subjects = (
        df[subject_cols]
        .drop_duplicates(subset=["subject"])
        .rename(columns={"subject": "subject_id", "project": "project_id"})
        .copy()
    )
    # NaN response (healthy controls) → NULL in SQLite
    subjects["response"] = subjects["response"].where(subjects["response"].notna(), other=None)
    subjects.to_sql("subjects", conn, if_exists="append", index=False)
    print(f"  subjects   : {len(subjects):>6,} rows inserted")

    # ── samples ───────────────────────────────────────────────────────────────
    sample_cols = ["sample", "subject", "sample_type", "time_from_treatment_start"]
    samples = (
        df[sample_cols]
        .drop_duplicates(subset=["sample"])
        .rename(columns={"sample": "sample_id", "subject": "subject_id"})
    )
    samples.to_sql("samples", conn, if_exists="append", index=False)
    print(f"  samples    : {len(samples):>6,} rows inserted")

    # ── cell_counts ───────────────────────────────────────────────────────────
    count_cols = ["sample", "b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]
    cell_counts = (
        df[count_cols]
        .drop_duplicates(subset=["sample"])
        .rename(columns={"sample": "sample_id"})
    )
    cell_counts.to_sql("cell_counts", conn, if_exists="append", index=False)
    print(f"  cell_counts: {len(cell_counts):>6,} rows inserted")


def verify(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for table in ("projects", "subjects", "samples", "cell_counts"):
        n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<15}: {n:>6,} rows")


def main() -> None:
    print(f"Target database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)
        load_csv(conn)
        print("\nVerification:")
        verify(conn)
        print(f"\nDatabase written to {DB_PATH}")
    finally:
        conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
