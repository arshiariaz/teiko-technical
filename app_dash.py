"""
app_dash.py — Teiko clinical trial dashboard.

Styled to Teiko's visual identity:
  Red    #E8312A  — non-responders, alerts, accent labels
  Teal   #2CB67D  — responders, positive outcomes (✓ color in their PDFs)
  Gray   #ABABAB  — display headings, secondary text
  Inter            — primary font

Run:
    python app_dash.py  →  http://127.0.0.1:8050
"""

import os
import sqlite3

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, dcc, html, dash_table
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------

RED    = "#E8312A"   # Teiko primary red
DARK   = "#6B7280"   # Medium gray — carcinoma, secondary category
GRAY   = "#ABABAB"   # Light gray — healthy, muted text
BODY   = "#555555"   # Body text
BG     = "#FFFFFF"   # Page background
CARD   = "#FAFAFA"   # Card background
BORDER = "#E8E8E8"   # Borders / dividers

# Single coherent accent — red only throughout
ACCENT_OVERVIEW = RED
ACCENT_STATS    = RED
ACCENT_SUBSET   = RED
ACCENT_BROWSER  = GRAY

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;"
    "0,600;0,700;1,400;1,500&display=swap"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "teiko.db")

CELL_POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]
POP_LABELS = {
    "b_cell":     "B Cell",
    "cd8_t_cell": "CD8 T Cell",
    "cd4_t_cell": "CD4 T Cell",
    "nk_cell":    "NK Cell",
    "monocyte":   "Monocyte",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

_cache: dict = {}


def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_summary() -> pd.DataFrame:
    if "summary" in _cache:
        return _cache["summary"]
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT sample_id AS sample, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte "
            "FROM cell_counts",
            conn,
        )
    long = df.melt(id_vars=["sample"], value_vars=CELL_POPULATIONS,
                   var_name="population", value_name="count")
    totals = long.groupby("sample")["count"].sum().rename("total_count")
    long = long.join(totals, on="sample")
    long["percentage"] = (long["count"] / long["total_count"] * 100).round(4)
    _cache["summary"] = long[["sample", "total_count", "population", "count", "percentage"]]
    return _cache["summary"]


def load_meta() -> pd.DataFrame:
    if "meta" in _cache:
        return _cache["meta"]
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT s.sample_id, sub.subject_id, sub.project_id,
                   sub.condition, sub.age, sub.sex,
                   sub.treatment, sub.response,
                   s.sample_type, s.time_from_treatment_start
            FROM samples s
            JOIN subjects sub ON s.subject_id = sub.subject_id
            """,
            conn,
        )
    _cache["meta"] = df
    return df


def load_stats_data() -> tuple:
    if "stats" in _cache:
        return _cache["stats"]
    summary = load_summary()
    meta = load_meta()
    filtered = meta[
        (meta["condition"] == "melanoma")
        & (meta["sample_type"] == "PBMC")
        & (meta["treatment"] == "miraclib")
        & (meta["response"].isin(["yes", "no"]))
    ]
    merged = summary.merge(
        filtered[["sample_id", "response"]],
        left_on="sample", right_on="sample_id",
    ).drop(columns="sample_id")

    rows = []
    for pop in CELL_POPULATIONS:
        sub = merged[merged["population"] == pop]
        r  = sub[sub["response"] == "yes"]["percentage"].dropna()
        nr = sub[sub["response"] == "no"]["percentage"].dropna()
        _, p = scipy_stats.mannwhitneyu(r, nr, alternative="two-sided")
        rows.append({
            "population":         POP_LABELS[pop],
            "mean_responders":    round(float(r.mean()), 2),
            "mean_nonresponders": round(float(nr.mean()), 2),
            "p_value":            round(float(p), 6),
            "significant":        p < 0.05,
        })
    stats_df = pd.DataFrame(rows)
    _cache["stats"] = (stats_df, merged)
    return stats_df, merged


def load_baseline() -> dict:
    if "baseline" in _cache:
        return _cache["baseline"]
    with get_conn() as conn:
        baseline = pd.read_sql_query(
            """
            SELECT s.sample_id, sub.subject_id, sub.project_id,
                   sub.sex, sub.response
            FROM samples s
            JOIN subjects sub ON s.subject_id = sub.subject_id
            WHERE sub.condition               = 'melanoma'
              AND s.sample_type               = 'PBMC'
              AND s.time_from_treatment_start = 0
              AND sub.treatment               = 'miraclib'
            """,
            conn,
        )
        avg_row = pd.read_sql_query(
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
        )
    avg_bcell = round(float(avg_row.iloc[0]["avg_b_cell"]), 2)
    subj = baseline.drop_duplicates("subject_id")
    _cache["baseline"] = {
        "by_project":  baseline.groupby("project_id")["sample_id"].count().reset_index()
                       .rename(columns={"project_id": "Project", "sample_id": "Sample Count"}),
        "by_response": subj.groupby("response")["subject_id"].count().reset_index()
                       .rename(columns={"response": "Response", "subject_id": "Subjects"}),
        "by_sex":      subj.groupby("sex")["subject_id"].count().reset_index()
                       .rename(columns={"sex": "Sex", "subject_id": "Subjects"}),
        "total_samples":  len(baseline),
        "total_subjects": len(subj),
        "avg_bcell_male_responders": avg_bcell,
    }
    return _cache["baseline"]


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _inter(size="13px", weight="400", color=BODY, italic=False, **extra):
    s = {"fontFamily": "Inter, sans-serif", "fontSize": size,
         "fontWeight": weight, "color": color}
    if italic:
        s["fontStyle"] = "italic"
    s.update(extra)
    return s


def accent_card(children, accent_color=RED, style=None):
    """Card with a colored left-border stripe, matching Teiko's panel cards."""
    base = {
        "background": BG,
        "border": f"1px solid {BORDER}",
        "borderLeft": f"4px solid {accent_color}",
        "borderRadius": "8px",
        "padding": "24px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 6px rgba(0,0,0,0.06)",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def plain_card(children, style=None):
    base = {
        "background": CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "20px 24px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.05)",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def section_header(label: str, title: str, subtitle: str = "", accent=RED):
    children = [
        html.Div(label, style=_inter("11px", "600", accent,
                                     letterSpacing="1.5px",
                                     textTransform="uppercase",
                                     marginBottom="6px")),
        html.H2(title, style={**_inter("28px", "700", GRAY), "margin": "0"}),
    ]
    if subtitle:
        children.append(
            html.P(subtitle, style=_inter("14px", "400", BODY, italic=True,
                                          marginTop="6px", marginBottom="0"))
        )
    return html.Div(children, style={"marginBottom": "28px"})


def kpi_card(value, label, color=RED, note=None):
    children = [
        html.Div(str(value), style=_inter("36px", "700", color, lineHeight="1")),
        html.Div(label, style=_inter("12px", "500", BODY, marginTop="6px")),
    ]
    if note:
        children.append(html.Div(note, style=_inter("11px", "400", GRAY,
                                                     marginTop="4px",
                                                     lineHeight="1.4")))
    return html.Div(children, style={
        "background": BG,
        "border": f"1px solid {BORDER}",
        "borderTop": f"3px solid {color}",
        "borderRadius": "8px",
        "padding": "20px 24px",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.05)",
        "textAlign": "center",
    })


def _chart_layout(fig, height=300, margin=None):
    m = margin or {"t": 20, "b": 40, "l": 50, "r": 20}
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font_family="Inter", font_color=BODY,
        yaxis={"gridcolor": BORDER, "zerolinecolor": BORDER},
        xaxis={"gridcolor": "rgba(0,0,0,0)"},
        height=height,
        margin=m,
    )
    return fig


def make_datatable(df: pd.DataFrame, table_id: str, page_size=15,
                   conditional=None):
    cond = [{"if": {"row_index": "odd"}, "backgroundColor": "#F7F7F7"}]
    if conditional:
        cond += conditional
    return dash_table.DataTable(
        id=table_id,
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": CARD, "fontWeight": "600",
            "fontFamily": "Inter, sans-serif", "fontSize": "12px",
            "color": BODY, "border": f"1px solid {BORDER}",
            "textTransform": "uppercase", "letterSpacing": "0.5px",
        },
        style_cell={
            "fontFamily": "Inter, sans-serif", "fontSize": "13px",
            "color": DARK, "border": f"1px solid {BORDER}",
            "padding": "9px 14px",
        },
        style_data_conditional=cond,
        sort_action="native",
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def build_sidebar(active: str):
    nav_items = [
        ("overview",   "Overview"),
        ("statistics", "Statistical Analysis"),
        ("subset",     "Subset Explorer"),
        ("browser",    "Data Browser"),
    ]

    # Teiko logo — angular red shape approximated with CSS clip-path
    logo = html.Div([
        html.Div(style={
            "width": "22px", "height": "22px",
            "background": RED,
            "clipPath": "polygon(0% 20%, 60% 0%, 100% 40%, 100% 100%, 40% 80%, 0% 100%)",
            "marginRight": "9px",
            "flexShrink": "0",
        }),
        html.Span("Teiko", style=_inter("19px", "700", DARK, letterSpacing="-0.4px")),
    ], style={"display": "flex", "alignItems": "center",
              "padding": "28px 18px 4px 18px"})

    tagline = html.Div(
        "Clinical Trial Dashboard",
        style=_inter("10px", "500", GRAY, letterSpacing="0.8px",
                     textTransform="uppercase", padding="0 18px 22px 18px"),
    )

    divider = html.Hr(style={"border": "none", "borderTop": f"1px solid {BORDER}",
                              "margin": "0 0 14px 0"})

    section_label = html.Div(
        "ANALYSIS",
        style=_inter("10px", "600", GRAY, letterSpacing="1.5px",
                     padding="0 18px 10px 18px"),
    )

    links = []
    for page_id, label in nav_items:
        is_active = page_id == active
        links.append(dcc.Link(
            label,
            href=f"/{page_id}",
            style={
                "display": "block",
                **_inter("13px", "600" if is_active else "400",
                         RED if is_active else BODY),
                "textDecoration": "none",
                "padding": "10px 18px",
                "backgroundColor": "rgba(232,49,42,0.06)" if is_active else "transparent",
                "borderLeft": f"3px solid {RED}" if is_active else "3px solid transparent",
                "marginBottom": "2px",
                "transition": "all 0.15s",
            },
        ))

    return html.Div([
        logo, tagline, divider, section_label,
        html.Div(links),
        html.Div(style={"flex": "1"}),
        html.Div("© Teiko Bio", style=_inter("10px", "400", GRAY,
                                             padding="24px 18px",
                                             borderTop=f"1px solid {BORDER}",
                                             marginTop="40px")),
    ], style={
        "width": "210px", "minHeight": "100vh",
        "background": BG, "borderRight": f"1px solid {BORDER}",
        "flexShrink": "0", "display": "flex", "flexDirection": "column",
    })


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview():
    meta    = load_meta()
    summary = load_summary()

    n_proj = meta["project_id"].nunique()
    n_subj = meta["subject_id"].nunique()
    n_samp = meta["sample_id"].nunique()

    # Donut — condition
    cond = meta.drop_duplicates("sample_id")["condition"].value_counts().reset_index()
    cond.columns = ["Condition", "Count"]
    fig_cond = px.pie(cond, values="Count", names="Condition", hole=0.5,
                      color_discrete_sequence=[RED, DARK, GRAY])
    _chart_layout(fig_cond, 270, {"t": 10, "b": 10, "l": 10, "r": 10})
    fig_cond.update_traces(textfont_size=12)

    # Donut — treatment
    treat = meta.drop_duplicates("sample_id")["treatment"].value_counts().reset_index()
    treat.columns = ["Treatment", "Count"]
    fig_treat = px.pie(treat, values="Count", names="Treatment", hole=0.5,
                       color_discrete_sequence=[RED, DARK, GRAY])
    _chart_layout(fig_treat, 270, {"t": 10, "b": 10, "l": 10, "r": 10})

    # Donut — sample type
    st = meta.drop_duplicates("sample_id")["sample_type"].value_counts().reset_index()
    st.columns = ["Sample Type", "Count"]
    fig_st = px.pie(st, values="Count", names="Sample Type", hole=0.5,
                    color_discrete_sequence=[RED, DARK])
    _chart_layout(fig_st, 270, {"t": 10, "b": 10, "l": 10, "r": 10})

    # Bar — mean population frequency across all samples
    avg = summary.groupby("population")["percentage"].mean().reset_index()
    avg["label"] = avg["population"].map(POP_LABELS)
    pop_colors = [RED, DARK, GRAY, BODY, "#888888"]
    fig_bar = go.Figure(go.Bar(
        x=avg["label"], y=avg["percentage"].round(2),
        marker_color=pop_colors,
        text=avg["percentage"].round(1).astype(str) + "%",
        textposition="outside",
        textfont={"family": "Inter", "size": 12},
    ))
    _chart_layout(fig_bar, 340, {"t": 40, "b": 40, "l": 50, "r": 20})
    fig_bar.update_layout(
        yaxis_title="Mean Relative Frequency (%)",
        showlegend=False,
        yaxis={"gridcolor": BORDER, "range": [0, fig_bar.data[0].y.max() * 1.2]},
    )

    # Stacked bar — sample counts by condition × time
    timeline = (
        meta.groupby(["condition", "time_from_treatment_start"])["sample_id"]
        .count().reset_index()
        .rename(columns={"time_from_treatment_start": "Day", "sample_id": "Samples"})
    )
    fig_time = px.bar(
        timeline, x="Day", y="Samples", color="condition", barmode="group",
        color_discrete_map={"melanoma": RED, "carcinoma": DARK, "healthy": GRAY},
    )
    _chart_layout(fig_time, 300)
    fig_time.update_layout(xaxis_title="Day", yaxis_title="Sample Count",
                           legend_title="Condition")

    return html.Div([
        section_header(
            "LOBLAW BIO — MIRACLIB TRIAL",
            "Study Overview",
            "Immune cell population analysis across all projects, conditions and time points.",
            ACCENT_OVERVIEW,
        ),

        # KPI row
        html.Div([
            html.Div(kpi_card(n_proj,                "Projects",         RED),  style={"flex": "1", "marginRight": "14px"}),
            html.Div(kpi_card(f"{n_subj:,}",         "Subjects",         DARK), style={"flex": "1", "marginRight": "14px"}),
            html.Div(kpi_card(f"{n_samp:,}",         "Samples",          RED),  style={"flex": "1", "marginRight": "14px"}),
            html.Div(kpi_card(len(CELL_POPULATIONS), "Cell Populations", DARK), style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "24px"}),

        # Donut trio
        html.Div([
            html.Div(accent_card([
                html.H4("Samples by Condition", style=_inter("13px", "600", BODY, marginBottom="10px")),
                dcc.Graph(figure=fig_cond, config={"displayModeBar": False}),
            ], ACCENT_OVERVIEW), style={"flex": "1", "marginRight": "14px"}),

            html.Div(accent_card([
                html.H4("Samples by Treatment", style=_inter("13px", "600", BODY, marginBottom="10px")),
                dcc.Graph(figure=fig_treat, config={"displayModeBar": False}),
            ], RED), style={"flex": "1", "marginRight": "14px"}),

            html.Div(accent_card([
                html.H4("Samples by Type", style=_inter("13px", "600", BODY, marginBottom="10px")),
                dcc.Graph(figure=fig_st, config={"displayModeBar": False}),
            ], DARK), style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "0"}),

        # Population bar
        accent_card([
            html.H4("Mean Population Frequency — All Samples",
                    style=_inter("13px", "600", BODY, marginBottom="4px")),
            html.P("Average relative frequency (%) of each immune cell population across the full dataset.",
                   style=_inter("12px", "400", GRAY, italic=True, marginBottom="12px")),
            dcc.Graph(figure=fig_bar, config={"displayModeBar": False}),
        ], ACCENT_SUBSET),

        # Timeline
        accent_card([
            html.H4("Samples per Condition over Time",
                    style=_inter("13px", "600", BODY, marginBottom="4px")),
            html.P("Number of samples collected at each timepoint (days from treatment start).",
                   style=_inter("12px", "400", GRAY, italic=True, marginBottom="12px")),
            dcc.Graph(figure=fig_time, config={"displayModeBar": False}),
        ], ACCENT_STATS),
    ])


# ---------------------------------------------------------------------------
# Page: Statistical Analysis
# ---------------------------------------------------------------------------

def page_statistics():
    stats_df, merged = load_stats_data()

    # Grouped boxplot — teal = responders, red = non-responders
    fig = go.Figure()
    for pop in CELL_POPULATIONS:
        label = POP_LABELS[pop]
        sub = merged[merged["population"] == pop]
        r  = sub[sub["response"] == "yes"]["percentage"]
        nr = sub[sub["response"] == "no"]["percentage"]

        fig.add_trace(go.Box(
            y=r, x=[label] * len(r),
            name="Responder", legendgroup="Responder",
            showlegend=(pop == CELL_POPULATIONS[0]),
            offsetgroup="R",
            marker_color=RED,
            line_color=RED,
            fillcolor="rgba(232,49,42,0.2)",
            boxpoints="outliers",
        ))
        fig.add_trace(go.Box(
            y=nr, x=[label] * len(nr),
            name="Non-Responder", legendgroup="Non-Responder",
            showlegend=(pop == CELL_POPULATIONS[0]),
            offsetgroup="NR",
            marker_color=GRAY,
            line_color=GRAY,
            fillcolor="rgba(171,171,171,0.2)",
            boxpoints="outliers",
        ))

    fig.update_layout(
        boxmode="group",
        paper_bgcolor=BG, plot_bgcolor=BG,
        font_family="Inter", font_color=BODY,
        yaxis_title="Relative Frequency (%)",
        xaxis_title="",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font={"size": 12, "family": "Inter"},
        ),
        height=500,
        margin={"t": 60, "b": 40, "l": 60, "r": 20},
        yaxis={"gridcolor": BORDER, "zerolinecolor": BORDER},
    )

    # Significance annotations
    for pop in CELL_POPULATIONS:
        label = POP_LABELS[pop]
        row = stats_df[stats_df["population"] == label]
        if not row.empty and row.iloc[0]["significant"]:
            pv = row.iloc[0]["p_value"]
            max_y = merged[merged["population"] == pop]["percentage"].max()
            fig.add_annotation(
                x=label, y=max_y * 1.04,
                text=f"✱ p = {pv:.4f}",
                showarrow=False,
                font={"size": 11, "color": RED, "family": "Inter"},
                yshift=12,
            )

    # Stats table
    display = stats_df.copy()
    display["significant"] = display["significant"].map({True: "Yes ✱", False: "No"})
    display.columns = ["Population", "Mean (Responders)", "Mean (Non-Resp.)", "p-value", "Significant"]

    stats_table = dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display.columns],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": CARD, "fontWeight": "600",
            "fontFamily": "Inter, sans-serif", "fontSize": "12px",
            "color": BODY, "border": f"1px solid {BORDER}",
            "textTransform": "uppercase", "letterSpacing": "0.5px",
        },
        style_cell={
            "fontFamily": "Inter, sans-serif", "fontSize": "13px",
            "color": DARK, "border": f"1px solid {BORDER}",
            "padding": "10px 16px", "textAlign": "center",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#F7F7F7"},
            {
                "if": {"filter_query": '{Significant} = "Yes ✱"'},
                "backgroundColor": "rgba(232,49,42,0.07)",
                "color": RED, "fontWeight": "600",
                "borderLeft": f"3px solid {RED}",
            },
        ],
    )

    sig_pops = stats_df[stats_df["significant"]]["population"].tolist()
    finding = (
        f"CD4 T Cell frequencies are significantly elevated in responders (p = "
        f"{stats_df[stats_df['population'] == 'CD4 T Cell']['p_value'].values[0]:.4f})."
        if sig_pops else "No populations reached significance at α = 0.05."
    )

    return html.Div([
        section_header(
            "MIRACLIB TRIAL — MELANOMA / PBMC",
            "Statistical Analysis",
            "Comparing responders vs non-responders using Mann-Whitney U (two-sided, α = 0.05).",
            ACCENT_STATS,
        ),

        # Legend key
        html.Div([
            html.Div([
                html.Div(style={"width": "12px", "height": "12px",
                                "background": RED, "borderRadius": "2px",
                                "marginRight": "6px", "display": "inline-block"}),
                html.Span("Responder", style=_inter("12px", "500", RED)),
            ], style={"display": "flex", "alignItems": "center", "marginRight": "20px"}),
            html.Div([
                html.Div(style={"width": "12px", "height": "12px",
                                "background": GRAY, "borderRadius": "2px",
                                "marginRight": "6px", "display": "inline-block"}),
                html.Span("Non-Responder", style=_inter("12px", "500", GRAY)),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={"display": "flex", "marginBottom": "20px"}),

        accent_card([
            html.H4("Population Frequencies: Responders vs Non-Responders",
                    style=_inter("14px", "600", DARK, marginBottom="4px")),
            html.P("Melanoma patients treated with miraclib, PBMC samples only.",
                   style=_inter("12px", "400", GRAY, italic=True, marginBottom="16px")),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ], ACCENT_STATS),

        # Key finding callout
        html.Div([
            html.Span("Key finding: ", style=_inter("13px", "700", DARK)),
            html.Span(finding, style=_inter("13px", "400", BODY)),
        ], style={
            "background": "rgba(44,182,125,0.08)",
            "border": f"1px solid {RED}",
            "borderLeft": f"4px solid {RED}",
            "borderRadius": "6px",
            "padding": "14px 18px",
            "marginBottom": "20px",
        }),

        accent_card([
            html.H4("Statistical Test Results",
                    style=_inter("14px", "600", DARK, marginBottom="4px")),
            html.P("Rows highlighted in red are statistically significant (p < 0.05).",
                   style=_inter("12px", "400", GRAY, italic=True, marginBottom="14px")),
            stats_table,
        ], RED),
    ])


# ---------------------------------------------------------------------------
# Page: Subset Explorer
# ---------------------------------------------------------------------------

def page_subset():
    data = load_baseline()

    def mini_bar(df, x_col, y_col):
        max_val = df[y_col].max()
        fig = go.Figure(go.Bar(
            x=df[x_col], y=df[y_col],
            marker_color=RED,
            text=df[y_col],
            textposition="outside",
            textfont={"family": "Inter", "size": 12},
        ))
        _chart_layout(fig, 240, {"t": 40, "b": 30, "l": 30, "r": 10})
        fig.update_layout(
            showlegend=False,
            yaxis={"gridcolor": BORDER, "range": [0, max_val * 1.25]},
        )
        return fig

    fig_proj = mini_bar(data["by_project"], "Project", "Sample Count")
    fig_resp = mini_bar(data["by_response"], "Response", "Subjects")
    fig_sex  = mini_bar(data["by_sex"], "Sex", "Subjects")

    def small_table(df):
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": CARD, "fontWeight": "600",
                          "fontSize": "11px", "fontFamily": "Inter",
                          "color": BODY, "border": f"1px solid {BORDER}"},
            style_cell={"fontFamily": "Inter", "fontSize": "12px",
                        "color": DARK, "textAlign": "center",
                        "padding": "7px 12px", "border": f"1px solid {BORDER}"},
        )

    return html.Div([
        section_header(
            "MIRACLIB — MELANOMA / PBMC / DAY 0",
            "Baseline Subset Explorer",
            "Filters: condition = melanoma · sample type = PBMC · time = 0 · treatment = miraclib",
            ACCENT_SUBSET,
        ),

        # KPI row
        html.Div([
            html.Div(kpi_card(data["total_samples"],  "Baseline Samples",   RED),
                     style={"flex": "1", "marginRight": "14px"}),
            html.Div(kpi_card(data["total_subjects"], "Unique Subjects",     DARK),
                     style={"flex": "1", "marginRight": "14px"}),
            html.Div(kpi_card(
                f"{data['avg_bcell_male_responders']:.2f}",
                "Avg B Cells",
                RED,
                note="Melanoma · Male · Responder · Day 0",
            ), style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "24px"}),

        # Three breakdown charts
        html.Div([
            html.Div(accent_card([
                html.H4("Samples per Project",
                        style=_inter("13px", "600", DARK, marginBottom="4px")),
                html.P("Number of baseline samples per study project.",
                       style=_inter("11px", "400", GRAY, italic=True, marginBottom="10px")),
                dcc.Graph(figure=fig_proj, config={"displayModeBar": False}),
                small_table(data["by_project"]),
            ], GRAY), style={"flex": "1", "marginRight": "14px"}),

            html.Div(accent_card([
                html.H4("Subjects by Response",
                        style=_inter("13px", "600", DARK, marginBottom="4px")),
                html.P("Responders (yes) vs non-responders (no).",
                       style=_inter("11px", "400", GRAY, italic=True, marginBottom="10px")),
                dcc.Graph(figure=fig_resp, config={"displayModeBar": False}),
                small_table(data["by_response"]),
            ], GRAY), style={"flex": "1", "marginRight": "14px"}),

            html.Div(accent_card([
                html.H4("Subjects by Sex",
                        style=_inter("13px", "600", DARK, marginBottom="4px")),
                html.P("Male vs female subject breakdown.",
                       style=_inter("11px", "400", GRAY, italic=True, marginBottom="10px")),
                dcc.Graph(figure=fig_sex, config={"displayModeBar": False}),
                small_table(data["by_sex"]),
            ], GRAY), style={"flex": "1"}),
        ], style={"display": "flex"}),
    ])


# ---------------------------------------------------------------------------
# Page: Data Browser
# ---------------------------------------------------------------------------

def page_browser():
    meta    = load_meta()
    summary = load_summary()
    merged  = meta.merge(summary, left_on="sample_id", right_on="sample", how="left").drop(columns="sample")

    conditions   = sorted(meta["condition"].dropna().unique())
    treatments   = sorted(meta["treatment"].dropna().unique())
    sample_types = sorted(meta["sample_type"].dropna().unique())
    projects     = sorted(meta["project_id"].dropna().unique())

    def dd(id_, opts, placeholder):
        return dcc.Dropdown(
            id=id_,
            options=[{"label": o, "value": o} for o in opts],
            multi=True,
            placeholder=placeholder,
            style={"fontSize": "13px", "fontFamily": "Inter"},
        )

    return html.Div([
        section_header(
            "FULL DATASET",
            "Data Browser",
            "Explore and filter all samples. One row per (sample, cell population).",
            ACCENT_BROWSER,
        ),

        plain_card([
            html.Div([
                html.Div([
                    html.Label("Condition", style=_inter("12px", "600", BODY, marginBottom="4px", display="block")),
                    dd("browser-condition", conditions, "All conditions"),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Treatment", style=_inter("12px", "600", BODY, marginBottom="4px", display="block")),
                    dd("browser-treatment", treatments, "All treatments"),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Sample Type", style=_inter("12px", "600", BODY, marginBottom="4px", display="block")),
                    dd("browser-sampletype", sample_types, "All types"),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Project", style=_inter("12px", "600", BODY, marginBottom="4px", display="block")),
                    dd("browser-project", projects, "All projects"),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "alignItems": "flex-end"}),
        ]),

        html.Div(id="browser-table-container"),
        dcc.Store(id="browser-full-data", data=merged.to_dict("records")),
    ])


# ---------------------------------------------------------------------------
# App + layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, GOOGLE_FONTS],
    suppress_callback_exceptions=True,
    title="Teiko — Clinical Trial Dashboard",
)
server = app.server  # exposed for gunicorn

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 0; background: #FFFFFF; }
  a { text-decoration: none !important; }
  .Select-control, .Select-menu-outer { font-family: Inter, sans-serif !important; }
</style>
</head>
<body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-wrapper", style={
        "display": "flex",
        "fontFamily": "Inter, sans-serif",
        "minHeight": "100vh",
        "background": BG,
    }),
], style={"margin": "0", "padding": "0"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("page-wrapper", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    routes = {
        "/overview":   ("overview",   page_overview),
        "/statistics": ("statistics", page_statistics),
        "/subset":     ("subset",     page_subset),
        "/browser":    ("browser",    page_browser),
    }
    if not pathname or pathname == "/" or pathname not in routes:
        key, fn = "overview", page_overview
    else:
        key, fn = routes[pathname]

    return [
        build_sidebar(key),
        html.Div(fn(), style={
            "flex": "1",
            "padding": "40px 44px",
            "background": BG,
            "maxWidth": "calc(100vw - 210px)",
            "overflowX": "hidden",
        }),
    ]


@app.callback(
    Output("browser-table-container", "children"),
    Input("browser-condition",  "value"),
    Input("browser-treatment",  "value"),
    Input("browser-sampletype", "value"),
    Input("browser-project",    "value"),
    Input("browser-full-data",  "data"),
)
def update_browser(conditions, treatments, sample_types, projects, data):
    df = pd.DataFrame(data)
    if conditions:   df = df[df["condition"].isin(conditions)]
    if treatments:   df = df[df["treatment"].isin(treatments)]
    if sample_types: df = df[df["sample_type"].isin(sample_types)]
    if projects:     df = df[df["project_id"].isin(projects)]

    cols = ["sample_id", "project_id", "subject_id", "condition",
            "treatment", "response", "sample_type",
            "time_from_treatment_start", "population", "count", "percentage"]
    df = df[[c for c in cols if c in df.columns]]

    return plain_card([
        html.Div(f"{len(df):,} rows",
                 style=_inter("12px", "400", GRAY, marginBottom="12px")),
        make_datatable(df.head(5000), "browser-table", page_size=20),
    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        raise SystemExit(
            f"Database not found at {DB_PATH}.\n"
            "Run `python load_data.py && python analysis.py` first."
        )
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
