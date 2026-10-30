"""Plot MMP attribution agreement and per-model squared correlations."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).parent
WISP_DIR = HERE / "explainability" / "wisp"
MMP_SCORE_DIR = WISP_DIR / "mmp_scores"
MODEL_SUMMARY_PATH = WISP_DIR / "summary.csv"
PLOT_PATH = HERE / "plots" / "explainability.png"
MIN_TEST_R2 = 0.35

MODEL_ORDER = (
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
)
MODEL_DISPLAY = {
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
}
MODEL_COLORS = dict(
    zip(
        MODEL_ORDER,
        (
            "#00441b",
            "#006d2c",
            "#238b45",
            "#41ab5d",
            "#74c476",
            "#a1d99b",
            "#005f73",
            "#0a9396",
            "#2a9d8f",
            "#52b788",
            "#6a994e",
            "#80b918",
            "#b5bd00",
            "#d4a017",
        ),
        strict=False,
    )
)


def _model_name(path: Path) -> str:
    return path.name.removesuffix(".csv.gz").removesuffix(".csv")


def _load_model_results(directory: Path, required_columns: set[str]) -> pd.DataFrame:
    paths = sorted(directory.glob("*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No result files found under {directory}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = required_columns - set(frame.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {missing_list}")
        frame["model"] = _model_name(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _load_test_r2() -> pd.Series:
    if not MODEL_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"No model summary found at {MODEL_SUMMARY_PATH}")

    summary = pd.read_csv(MODEL_SUMMARY_PATH)
    if list(summary.columns) != ["model", "test_r2"]:
        raise ValueError(f"{MODEL_SUMMARY_PATH} must contain only model and test_r2")
    if summary["model"].duplicated().any() or summary.isna().any().any():
        raise ValueError(f"{MODEL_SUMMARY_PATH} contains duplicate or missing values")
    return summary.set_index("model")["test_r2"].astype(float)


def _validate_shared_cohort(
    frame: pd.DataFrame,
    models: list[str],
    identifier: str,
) -> None:
    reference = set(frame.loc[frame["model"] == models[0], identifier])
    for model in models:
        model_identifiers = frame.loc[frame["model"] == model, identifier]
        if model_identifiers.duplicated().any():
            raise ValueError(f"Duplicate {identifier} values for {model}")
        if set(model_identifiers) != reference:
            raise ValueError(f"Inconsistent {identifier} cohort for {model}")


def load_results() -> tuple[pd.DataFrame, list[str]]:
    mmp_scores = _load_model_results(
        MMP_SCORE_DIR,
        {
            "pair_id",
            "prediction_delta",
            "whole_attribution_delta",
            "variable_attribution_delta",
        },
    ).dropna(
        subset=[
            "prediction_delta",
            "whole_attribution_delta",
            "variable_attribution_delta",
        ]
    )

    expected_models = set(MODEL_ORDER)
    mmp_models = set(mmp_scores["model"])
    if mmp_models != expected_models:
        raise ValueError(
            "MMP result models must match the configured models: "
            f"missing={sorted(expected_models - mmp_models)}, "
            f"unexpected={sorted(mmp_models - expected_models)}"
        )

    test_r2 = _load_test_r2()
    if set(test_r2.index) != expected_models:
        raise ValueError("Test-performance summaries must cover all configured models")

    models = [model for model in MODEL_ORDER if test_r2[model] >= MIN_TEST_R2]
    excluded = [model for model in MODEL_ORDER if model not in models]
    mmp_scores = mmp_scores.loc[mmp_scores["model"].isin(models)]
    _validate_shared_cohort(mmp_scores, models, "pair_id")
    excluded_text = ", ".join(
        f"{MODEL_DISPLAY[model]} ({test_r2[model]:.3f})" for model in excluded
    )
    print(f"Excluded models with test R² < {MIN_TEST_R2}: {excluded_text}")
    return mmp_scores, models


def plot_results(mmp_scores: pd.DataFrame, models: list[str]) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(
            "Whole-molecule attribution",
            "Variable-part attribution",
            "Attribution agreement by model (left: whole molecule; right: variable part)",
        ),
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
        row_heights=(0.55, 0.45),
    )

    for model in models:
        model_scores = mmp_scores.loc[mmp_scores["model"] == model]
        for column, attribution_column in (
            (1, "whole_attribution_delta"),
            (2, "variable_attribution_delta"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=model_scores[attribution_column],
                    y=model_scores["prediction_delta"],
                    mode="markers",
                    name=MODEL_DISPLAY[model],
                    legendgroup=model,
                    showlegend=False,
                    marker=dict(
                        color=MODEL_COLORS[model],
                        size=7,
                        opacity=0.68,
                    ),
                    customdata=model_scores[["pair_id"]],
                    hovertemplate=(
                        f"Model: {MODEL_DISPLAY[model]}<br>"
                        "Pair: %{customdata[0]}<br>"
                        "Attribution Δ: %{x:.3f}<br>"
                        "Prediction Δ: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )

    group_centers = [index * 2.4 for index in range(len(models))]
    bar_offset = 0.4
    bar_width = 0.72
    whole_r2 = []
    variable_r2 = []
    for model in models:
        model_scores = mmp_scores.loc[mmp_scores["model"] == model]
        prediction_delta = model_scores["prediction_delta"]
        whole_r2.append(
            float(prediction_delta.corr(model_scores["whole_attribution_delta"]) ** 2)
        )
        variable_r2.append(
            float(
                prediction_delta.corr(
                    model_scores["variable_attribution_delta"]
                )
                ** 2
            )
        )

    model_labels = [MODEL_DISPLAY[model] for model in models]
    bar_colors = [MODEL_COLORS[model] for model in models]
    for positions, values, edge_color, region in (
        (
            [center - bar_offset for center in group_centers],
            whole_r2,
            "#3f3f46",
            "Whole molecule",
        ),
        (
            [center + bar_offset for center in group_centers],
            variable_r2,
            "#d1d5db",
            "Variable part",
        ),
    ):
        fig.add_trace(
            go.Bar(
                x=positions,
                y=values,
                width=bar_width,
                marker=dict(
                    color=bar_colors,
                    line=dict(color=edge_color, width=3),
                ),
                showlegend=False,
                text=[f"{value:.2f}" for value in values],
                textposition="outside",
                textfont=dict(size=18),
                customdata=model_labels,
                hovertemplate=(
                    "%{customdata}<br>"
                    f"{region} R²: %{{y:.3f}}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    axis_style = dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickfont=dict(size=20),
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#777777",
        gridcolor="#e5e7eb",
        automargin=True,
    )
    fig.update_xaxes(
        title_text="Whole-molecule attribution Δ",
        title_font=dict(size=23),
        **axis_style,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Variable-part attribution Δ",
        title_font=dict(size=23),
        **axis_style,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="MMP prediction Δ",
        title_font=dict(size=23),
        **axis_style,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        matches="y",
        showticklabels=False,
        **axis_style,
        row=1,
        col=2,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=group_centers,
        ticktext=model_labels,
        range=[
            group_centers[0] - bar_offset - bar_width / 2 - 0.35,
            group_centers[-1] + bar_offset + bar_width / 2 + 0.35,
        ],
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickfont=dict(size=19),
        tickangle=-20,
        automargin=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="R²",
        title_font=dict(size=23),
        range=[0, 1.05],
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickfont=dict(size=20),
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#777777",
        gridcolor="#e5e7eb",
        automargin=True,
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=1100,
        margin=dict(l=90, r=40, t=100, b=80),
        font=dict(size=21, color="black"),
        showlegend=False,
    )
    panel_axes = (
        ("a)", "xaxis", "yaxis"),
        ("b)", "xaxis2", "yaxis2"),
        ("c)", "xaxis3", "yaxis3"),
    )
    for label, xaxis_name, yaxis_name in panel_axes:
        xaxis = getattr(fig.layout, xaxis_name)
        yaxis = getattr(fig.layout, yaxis_name)
        fig.add_annotation(
            text=f"<b>{label}</b>",
            x=xaxis.domain[0],
            y=yaxis.domain[1],
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=25, color="black"),
        )
    for annotation in fig.layout.annotations[:3]:
        annotation.font = dict(size=25, color="black")

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(PLOT_PATH), width=2400, height=1760, scale=2)
    print(f"Saved: {PLOT_PATH}")


def main() -> None:
    mmp_scores, models = load_results()
    plot_results(mmp_scores, models)


if __name__ == "__main__":
    main()
