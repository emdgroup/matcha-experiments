"""Collect mean MAE per model/endpoint and plot a 3×3 grid in summarize-CLI style."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent

DATASETS = {
    "caco2": ["CACO2_Papp_AB_endpoint", "CACO2_ER_endpoint"],
    "clint": ["HLM_CLint_endpoint", "MLM_CLint_endpoint"],
    "physchem": ["KSOL_endpoint", "LogD_endpoint"],
    "unbound": ["MBPB_endpoint", "MGMB_endpoint", "MPPB_endpoint"],
}

# Ordered by model family: tabular → SMILES → graph → graph3d
MODEL_ORDER = [
    "xgb_baseline",
    "mlp",
    "snn",
    "cnn",
    "rnn",
    "roformer",
    "attentivefp",
    "chemprop",
    "gatedgcn",
    "gin",
    "gps",
    "gt",
    "e3gnn",
    "gps3d",
    "gt3d",
    "finetuning-gatedgcn",
    "finetuning-roformer",
    "finetuning-b3lyp-gatedgcn",
]

# Family separators for axis tick styling
MODEL_FAMILIES = {
    "xgb_baseline": "Tabular",
    "mlp": "Tabular",
    "snn": "Tabular",
    "cnn": "SMILES",
    "rnn": "SMILES",
    "roformer": "SMILES",
    "attentivefp": "Graph",
    "chemprop": "Graph",
    "gatedgcn": "Graph",
    "gin": "Graph",
    "gps": "Graph",
    "gt": "Graph",
    "e3gnn": "Graph3D",
    "gps3d": "Graph3D",
    "gt3d": "Graph3D",
    "finetuning-gatedgcn": "Finetuning",
    "finetuning-roformer": "Finetuning",
    "finetuning-b3lyp-gatedgcn": "Finetuning",
}

MODEL_DISPLAY_NAMES = {
    "xgb_baseline": "XGBoost",
    "mlp": "MLP",
    "snn": "SNN",
    "cnn": "CNN",
    "rnn": "RNN",
    "roformer": "RoFormer",
    "attentivefp": "AttentiveFP",
    "chemprop": "Chemprop",
    "gatedgcn": "GatedGCN",
    "gin": "GIN",
    "gps": "GPS",
    "gt": "GT",
    "e3gnn": "E3GNN",
    "gps3d": "GPS3D",
    "gt3d": "GT3D",
    "finetuning-gatedgcn": "ADME-GatedGCN",
    "finetuning-roformer": "SSL-RoFormer",
    "finetuning-b3lyp-gatedgcn": "SSL-GatedGCN",
}

# Subplot layout – 3×3
SUBPLOTS = [
    ("caco2", "CACO2_Papp_AB_endpoint"),
    ("caco2", "CACO2_ER_endpoint"),
    ("clint", "HLM_CLint_endpoint"),
    ("clint", "MLM_CLint_endpoint"),
    ("physchem", "KSOL_endpoint"),
    ("physchem", "LogD_endpoint"),
    ("unbound", "MBPB_endpoint"),
    ("unbound", "MGMB_endpoint"),
    ("unbound", "MPPB_endpoint"),
]

# Colors matching summarize CLI
COLOR_BEST = "#149b5f"
COLOR_NOT_SIG = "#a5cd50"
COLOR_SIG = "#96d7d2"
COLOR_MISSING = "#d3d3d3"

SUBPLOT_TITLES = [
    "Caco-2 Papp A→B",
    "Caco-2 ER",
    "HLM CLint",
    "MLM CLint",
    "KSOL",
    "LogD",
    "MBPB",
    "MGMB",
    "MPPB",
]

ENDPOINT_LEADERBOARD = {
    "CACO2_Papp_AB_endpoint": "leaderboard_Caco-2_Permeability_Papp_AB_results.csv",
    "CACO2_ER_endpoint": "leaderboard_Caco-2_Permeability_Efflux_results.csv",
    "HLM_CLint_endpoint": "leaderboard_HLM_CLint_results.csv",
    "MLM_CLint_endpoint": "leaderboard_MLM_CLint_results.csv",
    "KSOL_endpoint": "leaderboard_KSOL_results.csv",
    "LogD_endpoint": "leaderboard_LogD_results.csv",
    "MBPB_endpoint": "leaderboard_MBPB_results.csv",
    "MGMB_endpoint": "leaderboard_MGMB_results.csv",
    "MPPB_endpoint": "leaderboard_MPPB_results.csv",
}

PERCENTILE_LINES = [
    (25, "25th challenge percentile", "#777777", "dot"),
    (50, "50th challenge percentile", "#777777", "dot"),
    (75, "75th challenge percentile", "#777777", "dot"),
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

# Generated with ``python eval/compress_performance.py compress``
PERFORMANCE = pd.read_csv(BASE / "eval" / "performance.csv", float_precision="round_trip")


def load_performance(dataset: str, model: str) -> dict | None:
    rows = PERFORMANCE[
        (PERFORMANCE["dataset"] == dataset)
        & (PERFORMANCE["model"] == model)
        & PERFORMANCE["sample"].isin(["mean", "std"])
    ]
    if rows.empty:
        return None
    return {
        sample: group.set_index("endpoint")[["MAE"]].to_dict("index")
        for sample, group in rows.groupby("sample")
    }


def load_pairwise(dataset: str) -> list[dict]:
    path = BASE / "summarize" / dataset / "summary_analysis.json"
    with open(path) as f:
        data = json.load(f)
    return data["pairwise_comparisons"]


def load_challenge_percentiles(endpoint: str) -> dict[int, float] | None:
    fname = ENDPOINT_LEADERBOARD.get(endpoint)
    if fname is None:
        return None
    path = BASE / "challenge_leaderboard" / fname
    if not path.exists():
        return None
    lb = pd.read_csv(path)
    maes = lb["MAE_display"].str.split(r"\s*±\s*", expand=True)[0].astype(float)
    return {pct: float(np.percentile(maes, pct)) for pct, *_ in PERCENTILE_LINES}


# ---------------------------------------------------------------------------
# Build per-endpoint MAE table
# ---------------------------------------------------------------------------

records = []
for dataset, endpoints in DATASETS.items():
    for model in MODEL_ORDER:
        perf = load_performance(dataset, model)
        if perf is None:
            continue
        mean = perf.get("mean", {})
        std = perf.get("std", {})
        for endpoint in endpoints:
            if endpoint not in mean:
                continue
            records.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "model": model,
                    "mae_mean": mean[endpoint]["MAE"],
                    "mae_std": std.get(endpoint, {}).get("MAE", 0.0),
                }
            )

df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Build significance lookup: (dataset, endpoint, model) → significance status
# "best" | "not_sig" | "sig" | "missing"
# ---------------------------------------------------------------------------

sig_status: dict[tuple, str] = {}

for dataset in DATASETS:
    pairwise = load_pairwise(dataset)
    for endpoint in DATASETS[dataset]:
        ep_data = df[(df["dataset"] == dataset) & (df["endpoint"] == endpoint)]
        if ep_data.empty:
            continue

        # Best model = lowest mean MAE among available models
        best_model = ep_data.loc[ep_data["mae_mean"].idxmin(), "model"]

        # Index comparisons for fast lookup
        comparisons: dict[frozenset, bool] = {}
        for comp in pairwise:
            if comp["endpoint"] != endpoint or comp["metric"] != "MAE":
                continue
            key = frozenset([comp["run_i_name"], comp["run_j_name"]])
            comparisons[key] = comp["significant_05"]

        for model in MODEL_ORDER:
            key = (dataset, endpoint, model)
            if ep_data[ep_data["model"] == model].empty:
                sig_status[key] = "missing"
            elif model == best_model:
                sig_status[key] = "best"
            else:
                pair_key = frozenset([best_model, model])
                is_sig = comparisons.get(pair_key, None)
                if is_sig is None:
                    sig_status[key] = "not_sig"
                elif is_sig:
                    sig_status[key] = "sig"
                else:
                    sig_status[key] = "not_sig"


def status_color(status: str) -> str:
    return {
        "best": COLOR_BEST,
        "not_sig": COLOR_NOT_SIG,
        "sig": COLOR_SIG,
        "missing": COLOR_MISSING,
    }[status]


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

H_SPACING = 0.03
V_SPACING = 0.06

fig = make_subplots(
    rows=3,
    cols=3,
    subplot_titles=SUBPLOT_TITLES,
    horizontal_spacing=H_SPACING,
    vertical_spacing=V_SPACING,
)

# We track which legend labels have been shown to avoid duplicates
legend_shown: set[str] = set()

LEGEND_LABELS = {
    "best": "Best model",
    "not_sig": "Not significantly different",
    "sig": "Significantly different",
}

for idx, (dataset, endpoint) in enumerate(SUBPLOTS):
    row = idx // 3 + 1
    col = idx % 3 + 1

    ep_data = df[(df["dataset"] == dataset) & (df["endpoint"] == endpoint)]

    # Group models by significance so each color gets one trace (for legend)
    groups: dict[str, dict] = {
        "best": {"x": [], "y": [], "err": []},
        "not_sig": {"x": [], "y": [], "err": []},
        "sig": {"x": [], "y": [], "err": []},
    }

    for model in MODEL_ORDER:
        status = sig_status.get((dataset, endpoint, model), "missing")
        if status == "missing":
            continue
        row_data = ep_data[ep_data["model"] == model]
        if row_data.empty:
            continue
        groups[status]["x"].append(float(row_data["mae_mean"].iloc[0]))
        groups[status]["y"].append(MODEL_DISPLAY_NAMES[model])
        groups[status]["err"].append(float(row_data["mae_std"].iloc[0]))

    for status, group in groups.items():
        if not group["x"]:
            continue
        label = LEGEND_LABELS[status]
        show = label not in legend_shown
        if show:
            legend_shown.add(label)
        color = status_color(status)
        fig.add_trace(
            go.Scatter(
                x=group["x"],
                y=group["y"],
                mode="markers",
                marker=dict(size=17, color=color, line=dict(width=1.5, color=color)),
                error_x=dict(
                    type="data",
                    array=group["err"],
                    visible=True,
                    color=color,
                    thickness=2.5,
                    width=8,
                ),
                name=label,
                showlegend=show,
                legendgroup=label,
            ),
            row=row,
            col=col,
        )

    # Challenge leaderboard percentile lines
    pctiles = load_challenge_percentiles(endpoint)
    if pctiles:
        for pct, label, color, dash in PERCENTILE_LINES:
            val = pctiles[pct]
            fig.add_vline(
                x=val,
                row=row,
                col=col,
                line_dash=dash,
                line_color=color,
                line_width=2,
                opacity=0.5,
            )

# ---------------------------------------------------------------------------
# Layout: shared y-axis with full model order, white background
# ---------------------------------------------------------------------------

fig.update_layout(
    width=1200,
    height=1600,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=32, color="black"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.04,
        xanchor="center",
        x=0.5,
        font=dict(size=32),
    ),
    margin=dict(l=100, r=40, t=160, b=80),
    title=dict(
        text="OpenADMET ExpansionRx Benchmark - MAE",
        x=0.5,
        xanchor="center",
        font=dict(size=56, color="black"),
    ),
)

# Fix y-axis to the full MODEL_ORDER (reversed so top = first)
display_order = [MODEL_DISPLAY_NAMES[m] for m in reversed(MODEL_ORDER)]

for i in range(1, 10):
    col = (i - 1) % 3 + 1
    axis_key = "yaxis" if i == 1 else f"yaxis{i}"
    fig.update_layout(
        **{
            axis_key: dict(
                categoryorder="array",
                categoryarray=display_order,
                showgrid=True,
                gridcolor="lightgray",
                gridwidth=1,
                tickfont=dict(size=35),
                showticklabels=col == 1,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            )
        }
    )
    axis_key_x = "xaxis" if i == 1 else f"xaxis{i}"
    fig.update_layout(
        **{
            axis_key_x: dict(
                showgrid=False,
                tickfont=dict(size=26),
                title=dict(text="", font=dict(size=26)),
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            )
        }
    )

fig.update_annotations(font=dict(size=46))

subplot_w = (1 - H_SPACING * 2) / 3
subplot_h = (1 - V_SPACING * 2) / 3
x_starts = [i * (subplot_w + H_SPACING) for i in range(3)]
y_tops = [1.0 - r * (subplot_h + V_SPACING) for r in range(3)]

for idx, label in enumerate("abcdefghi"):
    fig.add_annotation(
        text=f"<b>{label})</b>",
        x=x_starts[idx % 3],
        y=y_tops[idx // 3],
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=32, color="black"),
    )

out = BASE / "plots" / "eval_openadmet_mae.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.write_image(str(out), width=2400, height=3200, scale=3)
print(f"Saved: {out}")
