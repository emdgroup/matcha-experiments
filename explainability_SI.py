"""Plot mean descriptor importance with cross-explanation standard deviations."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from explainability import MIN_TEST_R2, MODEL_DISPLAY, MODEL_ORDER

HERE = Path(__file__).parent
WISP_DIR = HERE / "explainability" / "wisp"
DESCRIPTOR_SCORE_DIR = WISP_DIR / "descriptor_scores"
MODEL_SUMMARY_PATH = WISP_DIR / "summary.csv"
PLOT_PATH = HERE / "plots" / "explainability_SI.png"
TOP_DESCRIPTOR_COUNT = 10


def _model_name(path: Path) -> str:
    return path.name.removesuffix(".csv.gz").removesuffix(".csv")


def _load_test_r2() -> pd.Series:
    if not MODEL_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"No model summary found at {MODEL_SUMMARY_PATH}")

    summary = pd.read_csv(MODEL_SUMMARY_PATH)
    if list(summary.columns) != ["model", "test_r2"]:
        raise ValueError(f"{MODEL_SUMMARY_PATH} must contain only model and test_r2")
    if summary["model"].duplicated().any() or summary.isna().any().any():
        raise ValueError(f"{MODEL_SUMMARY_PATH} contains duplicate or missing values")
    return summary.set_index("model")["test_r2"].astype(float)


def load_descriptor_scores() -> pd.DataFrame:
    paths = sorted(DESCRIPTOR_SCORE_DIR.glob("*.csv.gz"))
    if not paths:
        raise FileNotFoundError(
            f"No descriptor result files found under {DESCRIPTOR_SCORE_DIR}"
        )

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required_columns = {
            "compound_id",
            "descriptor",
            "absolute_coefficient",
        }
        missing = required_columns - set(frame.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {missing_list}")
        frame["model"] = _model_name(path)
        frames.append(frame)

    descriptor_scores = pd.concat(frames, ignore_index=True).dropna(
        subset=["absolute_coefficient"]
    )
    test_r2 = _load_test_r2()
    expected_models = set(MODEL_ORDER)
    result_models = set(descriptor_scores["model"])
    if result_models != expected_models or set(test_r2.index) != expected_models:
        raise ValueError("Descriptor results and summaries must cover configured models")

    included_models = [model for model in MODEL_ORDER if test_r2[model] >= MIN_TEST_R2]
    excluded_models = [model for model in MODEL_ORDER if model not in included_models]
    descriptor_scores = descriptor_scores.loc[
        descriptor_scores["model"].isin(included_models)
    ]

    compound_cohorts = descriptor_scores[["model", "compound_id"]].drop_duplicates()
    reference_compounds = set(
        compound_cohorts.loc[
            compound_cohorts["model"] == included_models[0], "compound_id"
        ]
    )
    for model in included_models:
        model_compounds = set(
            compound_cohorts.loc[compound_cohorts["model"] == model, "compound_id"]
        )
        if model_compounds != reference_compounds:
            raise ValueError(f"Inconsistent compound cohort for {model}")

    if descriptor_scores.duplicated(["model", "compound_id", "descriptor"]).any():
        raise ValueError("Duplicate descriptor rows found for a model and compound")
    descriptor_sets = descriptor_scores.groupby(["model", "compound_id"])[
        "descriptor"
    ].agg(frozenset)
    expected_descriptors = descriptor_sets.iloc[0]
    if not descriptor_sets.map(lambda values: values == expected_descriptors).all():
        raise ValueError("Inconsistent descriptor sets across model-compound results")

    excluded_text = ", ".join(
        f"{MODEL_DISPLAY[model]} ({test_r2[model]:.3f})"
        for model in excluded_models
    )
    print(f"Excluded models with test R² < {MIN_TEST_R2}: {excluded_text}")
    return descriptor_scores


def plot_descriptors(descriptor_scores: pd.DataFrame) -> None:
    descriptor_summary = (
        descriptor_scores.groupby("descriptor")["absolute_coefficient"]
        .agg(["mean", "std"])
        .nlargest(TOP_DESCRIPTOR_COUNT, "mean")
        .sort_values("mean", ascending=False)
    )

    fig = go.Figure(
        go.Bar(
            x=descriptor_summary.index,
            y=descriptor_summary["mean"],
            marker=dict(
                color="#149b5f",
                line=dict(color="#0f6842", width=2),
            ),
            error_y=dict(
                type="data",
                array=descriptor_summary["std"],
                color="#0f6842",
                thickness=3.5,
                width=10,
            ),
            customdata=descriptor_summary[["std"]],
            hovertemplate=(
                "%{x}<br>Mean |coefficient|: %{y:.3f}<br>"
                "SD: %{customdata[0]:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Top 10 most important descriptors (mean ± SD)",
            x=0.5,
            xanchor="center",
            font=dict(size=28),
        ),
        width=1300,
        height=720,
        margin=dict(l=100, r=50, t=100, b=170),
        font=dict(size=21, color="black"),
        showlegend=False,
    )
    fig.update_xaxes(
        title_text="Descriptor",
        title_font=dict(size=24),
        tickfont=dict(size=18),
        tickangle=-25,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        automargin=True,
    )
    fig.update_yaxes(
        title_text="Mean absolute LIME coefficient",
        title_font=dict(size=24),
        tickfont=dict(size=20),
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#777777",
        gridcolor="#e5e7eb",
        automargin=True,
    )

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(PLOT_PATH), width=2600, height=1440, scale=2)
    print(f"Saved: {PLOT_PATH}")


def main() -> None:
    plot_descriptors(load_descriptor_scores())


if __name__ == "__main__":
    main()
