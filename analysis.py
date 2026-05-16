"""
analysis.py
-----------
Parts 2-4 of the Teiko clinical trial analysis pipeline.

Outputs written to outputs/:
    cell_frequency_summary.csv          – Part 2
    responder_vs_nonresponder_boxplot.png – Part 3
    statistical_results.csv             – Part 3
    baseline_samples_by_project.csv     – Part 4
    baseline_response_counts.csv        – Part 4
    baseline_sex_counts.csv             – Part 4

Run:
    python analysis.py
"""

import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "teiko.db")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

CELL_POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]
POP_LABELS = {
    "b_cell":      "B Cell",
    "cd8_t_cell":  "CD8 T Cell",
    "cd4_t_cell":  "CD4 T Cell",
    "nk_cell":     "NK Cell",
    "monocyte":    "Monocyte",
}

# Teiko brand colors
COLOR_RESPONDER     = "#E8312A"   # Teiko red → responders
COLOR_NON_RESPONDER = "#ABABAB"   # Teiko gray → non-responders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"{DB_PATH} not found – run `python load_data.py` first."
        )
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------------------------
# Part 2: Cell frequency summary table
# ---------------------------------------------------------------------------

def build_frequency_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Returns one row per (sample, population) with columns:
        sample, total_count, population, count, percentage
    """
    # Pull all cell counts (wide format from DB) and meta for context
    df = pd.read_sql_query(
        """
        SELECT
            cc.sample_id AS sample,
            cc.b_cell, cc.cd8_t_cell, cc.cd4_t_cell, cc.nk_cell, cc.monocyte
        FROM cell_counts cc
        ORDER BY cc.sample_id
        """,
        conn,
    )

    # Melt wide → long
    long_df = df.melt(
        id_vars=["sample"],
        value_vars=CELL_POPULATIONS,
        var_name="population",
        value_name="count",
    )

    # Compute total per sample and percentage
    totals = long_df.groupby("sample")["count"].sum().rename("total_count")
    long_df = long_df.join(totals, on="sample")
    long_df["percentage"] = (long_df["count"] / long_df["total_count"] * 100).round(4)

    return long_df[["sample", "total_count", "population", "count", "percentage"]].sort_values(
        ["sample", "population"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part 3: Statistical analysis – responders vs non-responders
# ---------------------------------------------------------------------------

def get_filtered_merged(conn: sqlite3.Connection, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to melanoma / PBMC / miraclib samples and merges with summary.
    """
    meta = pd.read_sql_query(
        """
        SELECT
            s.sample_id,
            sub.condition,
            sub.treatment,
            sub.response,
            s.sample_type
        FROM samples s
        JOIN subjects sub ON s.subject_id = sub.subject_id
        WHERE sub.condition   = 'melanoma'
          AND s.sample_type   = 'PBMC'
          AND sub.treatment   = 'miraclib'
          AND sub.response    IN ('yes', 'no')
        """,
        conn,
    )
    return summary.merge(
        meta[["sample_id", "response"]],
        left_on="sample",
        right_on="sample_id",
    ).drop(columns="sample_id")


def run_statistical_tests(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Mann-Whitney U test for each cell population.
    Returns DataFrame with population, means, p_value, significant.
    """
    rows = []
    for pop in CELL_POPULATIONS:
        pop_df = merged[merged["population"] == pop]
        resp     = pop_df[pop_df["response"] == "yes"]["percentage"].dropna()
        non_resp = pop_df[pop_df["response"] == "no"]["percentage"].dropna()

        _, p_value = scipy_stats.mannwhitneyu(resp, non_resp, alternative="two-sided")

        rows.append({
            "population":       POP_LABELS[pop],
            "mean_responders":  round(float(resp.mean()), 4),
            "mean_nonresponders": round(float(non_resp.mean()), 4),
            "p_value":          round(float(p_value), 6),
            "significant":      p_value < 0.05,
        })

    return pd.DataFrame(rows)


def plot_boxplots(merged: pd.DataFrame, stats_df: pd.DataFrame, out_path: str) -> None:
    """
    Grouped boxplot: one subplot per population, responders (red) vs non-responders (gray).
    Adds asterisk annotation when p < 0.05.
    """
    n_pops = len(CELL_POPULATIONS)
    fig, axes = plt.subplots(1, n_pops, figsize=(20, 7), sharey=False)
    fig.patch.set_facecolor("#FFFFFF")

    for ax, pop in zip(axes, CELL_POPULATIONS):
        pop_df = merged[merged["population"] == pop].copy()
        pop_df["Response"] = pop_df["response"].map({"yes": "Responder", "no": "Non-Responder"})

        sns.boxplot(
            data=pop_df,
            x="Response",
            y="percentage",
            hue="Response",
            palette={
                "Responder":     COLOR_RESPONDER,
                "Non-Responder": COLOR_NON_RESPONDER,
            },
            order=["Responder", "Non-Responder"],
            width=0.55,
            linewidth=1.2,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
            ax=ax,
            legend=False,
        )

        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, color="#E5E5E5")
        ax.set_title(POP_LABELS[pop], fontsize=11, fontweight="bold", color="#555555", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Relative Frequency (%)" if pop == CELL_POPULATIONS[0] else "", fontsize=9, color="#555555")
        ax.tick_params(axis="x", labelrotation=15, labelsize=8, colors="#555555")
        ax.tick_params(axis="y", labelsize=8, colors="#555555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#E5E5E5")

        # p-value annotation
        label = POP_LABELS[pop]
        row = stats_df[stats_df["population"] == label]
        if not row.empty and row.iloc[0]["significant"]:
            p = row.iloc[0]["p_value"]
            y_max = pop_df["percentage"].max()
            ax.annotate(
                f"* p={p:.4f}",
                xy=(0.5, 1.0),
                xycoords="axes fraction",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLOR_RESPONDER,
                fontweight="bold",
            )

    # Legend
    handles = [
        mpatches.Patch(facecolor=COLOR_RESPONDER,     label="Responder"),
        mpatches.Patch(facecolor=COLOR_NON_RESPONDER, label="Non-Responder"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.0),
    )

    fig.suptitle(
        "Immune Cell Population Frequencies: Responders vs Non-Responders\n"
        "Melanoma  ·  PBMC  ·  Miraclib",
        fontsize=13,
        fontweight="bold",
        color="#333333",
        y=1.06,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    print(f"  Boxplot saved → {out_path}")


# ---------------------------------------------------------------------------
# Part 4: Baseline subset analysis
# ---------------------------------------------------------------------------

def baseline_subset_analysis(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Melanoma / PBMC / miraclib / time=0 baseline cohort.

    Returns:
        by_project  – samples per project
        by_response – subjects by responder status
        by_sex      – subjects by sex
    """
    baseline = pd.read_sql_query(
        """
        SELECT
            s.sample_id,
            sub.subject_id,
            sub.project_id,
            sub.sex,
            sub.response
        FROM samples s
        JOIN subjects sub ON s.subject_id = sub.subject_id
        WHERE sub.condition               = 'melanoma'
          AND s.sample_type               = 'PBMC'
          AND s.time_from_treatment_start = 0
          AND sub.treatment               = 'miraclib'
        """,
        conn,
    )

    by_project = (
        baseline.groupby("project_id")["sample_id"]
        .count()
        .reset_index()
        .rename(columns={"project_id": "project", "sample_id": "sample_count"})
    )

    subjects = baseline.drop_duplicates("subject_id")

    by_response = (
        subjects.groupby("response")["subject_id"]
        .count()
        .reset_index()
        .rename(columns={"response": "response_status", "subject_id": "subject_count"})
    )

    by_sex = (
        subjects.groupby("sex")["subject_id"]
        .count()
        .reset_index()
        .rename(columns={"subject_id": "subject_count"})
    )

    # Avg B cells: melanoma males, miraclib, responders, time=0
    avg_bcell = pd.read_sql_query(
        """
        SELECT AVG(cc.b_cell) AS avg_b_cell
        FROM samples s
        JOIN subjects sub ON s.subject_id = sub.subject_id
        JOIN cell_counts cc ON s.sample_id = cc.sample_id
        WHERE sub.condition               = 'melanoma'
          AND s.sample_type               = 'PBMC'
          AND s.time_from_treatment_start = 0
          AND sub.treatment               = 'miraclib'
          AND sub.response                = 'yes'
          AND sub.sex                     = 'M'
        """,
        conn,
    ).iloc[0]["avg_b_cell"]

    return by_project, by_response, by_sex, round(avg_bcell, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    conn = get_conn()

    print("=" * 60)
    print("PART 2 – Cell Frequency Summary Table")
    print("=" * 60)
    summary = build_frequency_summary(conn)
    summary_path = os.path.join(OUT_DIR, "cell_frequency_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  {len(summary):,} rows → {summary_path}")

    print()
    print("=" * 60)
    print("PART 3 – Statistical Analysis (Melanoma / PBMC / Miraclib)")
    print("=" * 60)
    merged = get_filtered_merged(conn, summary)
    stats_df = run_statistical_tests(merged)

    stats_path = os.path.join(OUT_DIR, "statistical_results.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"  Results → {stats_path}")
    print()
    print(stats_df.to_string(index=False))

    sig = stats_df[stats_df["significant"] == True]["population"].tolist()
    print()
    if sig:
        print(f"  Significant populations (p < 0.05): {', '.join(sig)}")
    else:
        print("  No populations reached significance at p < 0.05.")

    plot_path = os.path.join(OUT_DIR, "responder_vs_nonresponder_boxplot.png")
    plot_boxplots(merged, stats_df, plot_path)

    print()
    print("=" * 60)
    print("PART 4 – Baseline Subset Analysis")
    print("=" * 60)
    by_project, by_response, by_sex, avg_bcell = baseline_subset_analysis(conn)

    proj_path   = os.path.join(OUT_DIR, "baseline_samples_by_project.csv")
    resp_path   = os.path.join(OUT_DIR, "baseline_response_counts.csv")
    sex_path    = os.path.join(OUT_DIR, "baseline_sex_counts.csv")
    bcell_path  = os.path.join(OUT_DIR, "avg_bcell_male_responders.csv")

    by_project.to_csv(proj_path, index=False)
    by_response.to_csv(resp_path, index=False)
    by_sex.to_csv(sex_path, index=False)
    pd.DataFrame([{"group": "Melanoma Male Responders (time=0)", "avg_b_cell": avg_bcell}]).to_csv(bcell_path, index=False)

    print("\n  Samples per project:")
    print(by_project.to_string(index=False, col_space=12))
    print("\n  Subjects by response status:")
    print(by_response.to_string(index=False, col_space=16))
    print("\n  Subjects by sex:")
    print(by_sex.to_string(index=False, col_space=14))
    print(f"\n  Avg B cells (melanoma males, miraclib, responders, time=0): {avg_bcell:.2f}")
    print(f"\n  Outputs: {proj_path}")
    print(f"           {resp_path}")
    print(f"           {sex_path}")
    print(f"           {bcell_path}")

    conn.close()
    print()
    print("Analysis complete.")


if __name__ == "__main__":
    main()
