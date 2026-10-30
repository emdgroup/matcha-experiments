"""Boxplot summary of the uncertainty-quantification benchmark.

Reads ``uncertainty/logging_df.csv`` produced by
``uncertainty/run_benchmark.py`` and renders one boxplot per metric, with each
box showing the distribution across all (model, endpoint) combos for a given
UQ approach.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).parent
LOG_PATH = HERE / "uncertainty" / "logging_df.csv"
SHARD_DIR = HERE / "uncertainty" / "logging_shards"
PLOT_PATH = HERE / "plots" / "uq_benchmark_comparison.png"

APPROACHES = ("mc_dropout", "mve", "ensemble", "error_model")
APPROACH_DISPLAY = {
    "mc_dropout": "MC Dropout",
    "mve": "MVE",
    "ensemble": "Ensemble",
    "error_model": "Error Model",
}
APPROACH_COLORS = {
    "mc_dropout": "lightgrey",
    "mve": "#96d7d2",
    "ensemble": "#b4dc96",
    "error_model": "#149b5f",
}

METRICS = [
    ("top10_mae", "MAE — top 10% most confident", "MAE"),
    ("bottom10_mae", "MAE — bottom 10% least confident", "MAE"),
    ("spearman", "Spearman R (|residual| vs. uncertainty)", "Spearman R"),
]


def plot_boxplots(df: pd.DataFrame) -> None:
    fig = make_subplots(
        rows=1,
        cols=len(METRICS),
        subplot_titles=[title for _, title, _ in METRICS],
        horizontal_spacing=0.08,
    )

    for col, (metric, _title, y_label) in enumerate(METRICS, start=1):
        for approach in APPROACHES:
            values = df.loc[df["approach"] == approach, metric].to_numpy()
            fig.add_trace(
                go.Box(
                    y=values,
                    name=APPROACH_DISPLAY[approach],
                    marker_color=APPROACH_COLORS[approach],
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    legendgroup=approach,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

    axis_style = dict(
        tickfont=dict(size=22),
        automargin=True,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )
    for col, (_metric, _title, y_label) in enumerate(METRICS, start=1):
        fig.update_yaxes(title_text=y_label, title_font=dict(size=26), **axis_style, row=1, col=col)
        fig.update_xaxes(showticklabels=False, **axis_style, row=1, col=col)

    fig.update_layout(
        template="plotly_white",
        width=1400,
        height=520,
        margin=dict(l=100, r=40, t=90, b=140),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=24),
        ),
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=24)

    for index, label in enumerate("abc", start=1):
        axis_suffix = "" if index == 1 else str(index)
        xaxis = getattr(fig.layout, f"xaxis{axis_suffix}")
        yaxis = getattr(fig.layout, f"yaxis{axis_suffix}")
        fig.add_annotation(
            text=f"<b>{label})</b>",
            x=xaxis.domain[0],
            y=yaxis.domain[1],
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=24, color="black"),
        )

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(PLOT_PATH), width=2100, height=800, scale=2)
    print(f"Saved: {PLOT_PATH}")


def _load_results() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    shards = sorted(SHARD_DIR.glob("task_*.csv"))
    if shards:
        return pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    raise FileNotFoundError(
        f"No results found — expected {LOG_PATH} or shards under {SHARD_DIR}. "
        "Run `python uncertainty/run_benchmark.py` first."
    )


def main() -> None:
    df = _load_results()
    plot_boxplots(df)


if __name__ == "__main__":
    main()
