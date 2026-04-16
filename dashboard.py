import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS_PATH = Path("results.json")
HYBRID_NAME = "Hybrid PSO-GWO"
CHART_FILES = [
    "plot1_f1_score.png",
    "plot2_fairness.png",
    "plot3_cost.png",
    "plot4_auc_bar.png",
]

ALGO_COLORS = {
    "AllClient": "#6b7280",
    "Random": "#f59e0b",
    "Greedy": "#2563eb",
    "Clustered": "#9333ea",
    "PSO": "#fb923c",
    "GWO": "#2dd4bf",
    "Hybrid PSO-GWO": "#10b981",
}


def inject_theme():
    st.markdown(
        """
<style>
:root {
    --bg-color: #0b0f19;
    --panel-color: rgba(20, 27, 45, 0.7);
    --panel-border: rgba(255, 255, 255, 0.08);
    --text-main: #f3f4f6;
    --text-muted: #9ca3af;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.4);
    --good: #10b981;
    --good-bg: rgba(16, 185, 129, 0.15);
    --bad: #ef4444;
    --bad-bg: rgba(239, 68, 68, 0.15);
}

/* Force dark background for Streamlit app */
.stApp {
    background: radial-gradient(circle at 15% 10%, rgba(30, 58, 138, 0.4) 0%, transparent 40%),
                radial-gradient(circle at 85% 90%, rgba(16, 185, 129, 0.15) 0%, transparent 40%),
                var(--bg-color);
    color: var(--text-main);
}

/* Override default sidebar */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid var(--panel-border) !important;
}

[data-testid="stSidebar"] .stSlider > div > div > div > div {
    color: var(--text-main) !important;
}

/* Global Typography Fixes */
.stMarkdown p, .stMarkdown li, h1, h2, h3, h4, h5, h6, .st-emotion-cache-1629p8f h1, .st-emotion-cache-1629p8f h2, .st-emotion-cache-1629p8f h3 {
    color: var(--text-main) !important;
}

.hero {
    padding: 2rem 2.5rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.8), rgba(59, 130, 246, 0.6));
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3), inset 0 0 20px rgba(255, 255, 255, 0.05);
    color: #ffffff !important;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}

.hero h1 {
    margin: 0 !important;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    color: #ffffff !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

.hero p {
    margin: 0.8rem 0 0 0 !important;
    opacity: 0.9 !important;
    font-size: 1.15rem !important;
    font-weight: 300 !important;
    color: #e0e7ff !important;
}

.section-card {
    background: var(--panel-color);
    border: 1px solid var(--panel-border);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1rem;
}
.section-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.kpi {
    background: var(--panel-color);
    border: 1px solid var(--panel-border);
    border-left: 4px solid var(--accent);
    border-radius: 16px;
    padding: 1.2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.kpi:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px var(--accent-glow);
    border-left: 4px solid #60a5fa;
}

.kpi .label {
    color: var(--text-muted);
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.4rem;
}

.kpi .value {
    color: var(--text-main);
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1.2;
}

.badge-pass {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 9999px;
    color: var(--good);
    background: var(--good-bg);
    border: 1px solid rgba(16, 185, 129, 0.3);
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.1);
}

.badge-fail {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 9999px;
    color: var(--bad);
    background: var(--bad-bg);
    border: 1px solid rgba(239, 68, 68, 0.3);
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.1);
}

hr {
    border-color: rgba(255, 255, 255, 0.1) !important;
}

.small-note {
    color: #6b7280;
    font-size: 0.86rem;
}

@media (max-width: 900px) {
    .hero h1 { font-size: 1.6rem !important; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def status_badge(value):
    cls = "badge-pass" if value == "PASS" else "badge-fail"
    return f"<span class='{cls}'>{value}</span>"


def render_kpi_card(label, value):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summarize_metrics(metrics, fairness_weight=0.2, cost_weight=0.1):
    f1 = np.array(metrics["f1"], dtype=float)
    f1_var = np.array(metrics["f1_var"], dtype=float)
    cost = np.array(metrics["cost"], dtype=float)
    auc = np.array(metrics["auc"], dtype=float)

    rounds = np.arange(1, len(f1) + 1, dtype=float)
    f1_slope = float(np.polyfit(rounds, f1, 1)[0]) if len(f1) > 1 else 0.0
    auc_slope = float(np.polyfit(rounds, auc, 1)[0]) if len(auc) > 1 else 0.0

    utility = float(np.mean(f1 - fairness_weight * f1_var - cost_weight * cost))

    return {
        "mean_f1": float(np.mean(f1)),
        "final_f1": float(f1[-1]),
        "best_f1": float(np.max(f1)),
        "worst_f1": float(np.min(f1)),
        "f1_std": float(np.std(f1)),
        "f1_gain": float(f1[-1] - f1[0]) if len(f1) > 1 else 0.0,
        "mean_auc": float(np.mean(auc)),
        "final_auc": float(auc[-1]),
        "auc_std": float(np.std(auc)),
        "auc_gain": float(auc[-1] - auc[0]) if len(auc) > 1 else 0.0,
        "mean_var": float(np.mean(f1_var)),
        "mean_cost": float(np.mean(cost)),
        "cost_std": float(np.std(cost)),
        "efficiency": float(np.mean(f1 / (cost + 1e-12))),
        "utility": utility,
        "f1_slope": f1_slope,
        "auc_slope": auc_slope,
    }


def load_results():
    if not RESULTS_PATH.exists():
        return None
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_timeseries_df(results):
    rows = []
    for algo, m in results.items():
        rounds = len(m["f1"])
        for r in range(rounds):
            rows.append(
                {
                    "algorithm": algo,
                    "round": r + 1,
                    "f1": float(m["f1"][r]),
                    "f1_var": float(m["f1_var"][r]),
                    "cost": float(m["cost"][r]),
                    "auc": float(m["auc"][r]),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["round_label"] = "Round " + df["round"].astype(str)
        df = df.sort_values(["algorithm", "round"]).reset_index(drop=True)
        for metric in ["f1", "f1_var", "cost", "auc"]:
            df[f"{metric}_roll3"] = (
                df.groupby("algorithm")[metric]
                .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            )
    return df


def build_round_summary(ts_df, metric):
    summary = ts_df.groupby("round")[metric].agg(["mean", "min", "max", "std"]).reset_index()
    summary.columns = ["round", "mean", "min", "max", "std"]
    summary["round_label"] = "Round " + summary["round"].astype(str)
    return summary


def build_metric_matrix(ts_df, metric, selected_algorithms):
    subset = ts_df[ts_df["algorithm"].isin(selected_algorithms)]
    return subset.pivot(index="algorithm", columns="round", values=metric).reindex(selected_algorithms)


def compute_selected_stats(ts_df, selected_algorithms, metric, start_round, end_round):
    subset = ts_df[
        ts_df["algorithm"].isin(selected_algorithms)
        & (ts_df["round"] >= start_round)
        & (ts_df["round"] <= end_round)
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    grouped = subset.groupby("algorithm")
    stats = grouped[metric].agg(["mean", "min", "max", "std", "first", "last"]).reset_index()
    stats["delta"] = stats["last"] - stats["first"]
    stats["rounds"] = grouped.size().values
    stats = stats.rename(
        columns={
            "mean": f"{metric}_mean",
            "min": f"{metric}_min",
            "max": f"{metric}_max",
            "std": f"{metric}_std",
            "first": f"{metric}_first",
            "last": f"{metric}_last",
            "delta": f"{metric}_delta",
        }
    )
    return stats


def render_interactive_trend(ts_df, metric, selected_algorithms, start_round, end_round, show_rolling=False):
    subset = ts_df[
        ts_df["algorithm"].isin(selected_algorithms)
        & (ts_df["round"] >= start_round)
        & (ts_df["round"] <= end_round)
    ].copy()

    if subset.empty:
        st.warning("No data available for the selected filters.")
        return

    value_column = f"{metric}_roll3" if show_rolling else metric
    chart_df = subset.copy()
    chart_df["display_value"] = chart_df[value_column]

    fig = px.line(
        chart_df,
        x="round",
        y="display_value",
        color="algorithm",
        markers=True,
        color_discrete_map=ALGO_COLORS,
        title=f"{metric.upper()} vs Round",
    )
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate="Algorithm: %{fullData.name}<br>Round: %{x}<br>Value: %{y:.4f}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title_text="Algorithm",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Communication Round",
        yaxis_title=metric.upper(),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_metric_summary_cards(summary, selected_algorithms):
    if not selected_algorithms:
        return

    cols = st.columns(min(4, len(selected_algorithms)))
    for idx, algo in enumerate(selected_algorithms[: len(cols)]):
        values = summary[algo]
        with cols[idx]:
            render_kpi_card(
                algo,
                f"F1 {values['mean_f1']:.3f} | AUC {values['mean_auc']:.3f} | Utility {values['utility']:.3f}",
            )


def compute_validation(summary):
    hybrid = summary[HYBRID_NAME]
    baselines = {k: v for k, v in summary.items() if k != HYBRID_NAME}

    checks = {
        "mean_f1_beats_all_baselines": hybrid["mean_f1"] > max(v["mean_f1"] for v in baselines.values()),
        "best_round_f1_beats_all_baselines": hybrid["best_f1"] > max(v["best_f1"] for v in baselines.values()),
        "selection_efficiency_beats_all_baselines": hybrid["efficiency"] > max(v["efficiency"] for v in baselines.values()),
        "overall_utility_beats_all_baselines": hybrid["utility"] > max(v["utility"] for v in baselines.values()),
    }
    return checks


def comparison_table(summary):
    df = pd.DataFrame(summary).T
    return df.sort_values("utility", ascending=False)


def explain_hybrid_advantage(table):
    rank = table.index.tolist()
    hybrid_rank = rank.index(HYBRID_NAME) + 1

    hybrid = table.loc[HYBRID_NAME]
    best_baseline = table[table.index != HYBRID_NAME].iloc[0]

    explanation = []
    explanation.append(
        f"Hybrid ranking by utility: {hybrid_rank}/{len(rank)} (higher utility is better)."
    )
    explanation.append(
        f"Hybrid mean F1: {hybrid['mean_f1']:.4f} vs best baseline: {best_baseline['mean_f1']:.4f}."
    )
    explanation.append(
        f"Hybrid best-round F1: {hybrid['best_f1']:.4f} vs best baseline: {best_baseline['best_f1']:.4f}."
    )
    explanation.append(
        f"Hybrid client-selection efficiency (F1/cost): {hybrid['efficiency']:.4f} vs best baseline: {best_baseline['efficiency']:.4f}."
    )
    explanation.append(
        f"Hybrid mean selection cost: {hybrid['mean_cost']:.4f} (lower means fewer clients selected per round)."
    )
    explanation.append(
        f"Hybrid fairness variance: {hybrid['mean_var']:.6f} (lower indicates more balanced client quality across selected clients)."
    )
    return explanation


def main():
    st.set_page_config(page_title="Hybrid FL Dashboard", layout="wide", initial_sidebar_state="expanded")
    inject_theme()

    st.markdown(
        """
        <div class="hero">
            <h1>Hybrid PSO-GWO Federated IDS Dashboard</h1>
            <p>End-to-end project intelligence: model overview, optimization flow, and evidence-backed baseline comparison.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results = load_results()
    if results is None:
        st.error("results.json not found. Run main.py first.")
        st.stop()

    if HYBRID_NAME not in results:
        st.error(f"{HYBRID_NAME} is missing in results.json.")
        st.stop()

    st.sidebar.header("Dashboard Controls")
    fairness_weight = st.sidebar.slider("Utility fairness weight", 0.0, 1.0, 0.2, 0.05)
    cost_weight = st.sidebar.slider("Utility cost weight", 0.0, 1.0, 0.1, 0.05)
    max_rounds = len(next(iter(results.values()))["f1"])

    summary = {
        name: summarize_metrics(metrics, fairness_weight=fairness_weight, cost_weight=cost_weight)
        for name, metrics in results.items()
    }
    checks = compute_validation(summary)
    table = comparison_table(summary)
    ts_df = build_timeseries_df(results)

    hybrid = table.loc[HYBRID_NAME]
    overall = "PASS" if all(checks.values()) else "FAIL"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Hybrid Mean F1", f"{hybrid['mean_f1']:.4f}")
    with k2:
        render_kpi_card("Hybrid Best F1", f"{hybrid['best_f1']:.4f}")
    with k3:
        render_kpi_card("Hybrid Efficiency", f"{hybrid['efficiency']:.4f}")
    with k4:
        render_kpi_card("Validation Status", overall)

    st.sidebar.subheader("Interactive Filters")
    all_algorithms = list(results.keys())
    default_algorithms = [HYBRID_NAME] + [a for a in all_algorithms if a != HYBRID_NAME][:3]
    selected_algorithms = st.sidebar.multiselect(
        "Algorithms to compare",
        all_algorithms,
        default=default_algorithms,
    )
    metric = st.sidebar.selectbox("Metric", ["f1", "f1_var", "cost", "auc"], index=0)
    round_range = st.sidebar.slider("Round range", 1, max_rounds, (1, max_rounds))
    show_rolling = st.sidebar.toggle("Show 3-round rolling average", value=False)
    selected_round = st.sidebar.slider("Inspect round", 1, max_rounds, min(max_rounds, round_range[1]))

    if not selected_algorithms:
        selected_algorithms = default_algorithms

    intro_col1, intro_col2 = st.columns([2, 1])
    with intro_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Project Introduction")
        st.markdown(
            """
This project builds a federated Intrusion Detection System (IDS) over heterogeneous IoT clients.
Each client trains locally on private data, and the server aggregates selected client updates each round.
The key contribution is a Hybrid PSO-GWO selector that learns which clients to include and how to weight their updates.

Why this matters:
- Better global detection quality (F1 and AUC)
- Lower communication load (fewer clients selected)
- Better fairness stability (lower selected-client performance variance)
"""
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with intro_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Current Validation")
        for key, ok in checks.items():
            v = "PASS" if ok else "FAIL"
            st.markdown(f"<div><b>{key}</b><br>{status_badge(v)}</div>", unsafe_allow_html=True)
            st.write("")
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.markdown(f"<b>overall_result</b><br>{status_badge(overall)}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Hybrid vs Baselines", "Comparison Charts", "Model Flow", "Raw Metrics"]
    )

    with tab1:
        st.subheader("Why Hybrid Beats Baselines")
        render_metric_summary_cards(summary, selected_algorithms)

        styled = table.style.format("{:.4f}")
        st.dataframe(styled, use_container_width=True)

        for line in explain_hybrid_advantage(table):
            st.write(f"- {line}")

        st.markdown("Key interpretation:")
        st.markdown(
            """
Hybrid outperforms by balancing exploration (PSO dynamics) and exploitation (GWO leader guidance).
This typically produces better client subsets with higher information value per communication cost.
"""
        )

        st.subheader("Interactive Metric Trends")
        render_interactive_trend(
            ts_df,
            metric,
            selected_algorithms,
            round_range[0],
            round_range[1],
            show_rolling=show_rolling,
        )

        round_metrics = ts_df[
            (ts_df["round"] == selected_round) & (ts_df["algorithm"].isin(selected_algorithms))
        ][["algorithm", "f1", "f1_var", "cost", "auc"]].set_index("algorithm")
        st.markdown(f"#### Round {selected_round} snapshot")
        st.dataframe(round_metrics.style.format("{:.4f}"), use_container_width=True)

        comparison_metric = st.selectbox("Round drilldown metric", ["f1", "f1_var", "cost", "auc"], index=0, key="round-drilldown")
        selected_stats = compute_selected_stats(ts_df, selected_algorithms, comparison_metric, round_range[0], round_range[1])
        if not selected_stats.empty:
            st.markdown(f"#### {comparison_metric.upper()} statistics in selected range")
            numeric_columns = selected_stats.select_dtypes(include=[np.number]).columns
            st.dataframe(
                selected_stats.style.format("{:.4f}", subset=pd.IndexSlice[:, numeric_columns]),
                use_container_width=True,
            )

            chart_df = build_round_summary(
                ts_df[
                    ts_df["algorithm"].isin(selected_algorithms)
                    & (ts_df["round"] >= round_range[0])
                    & (ts_df["round"] <= round_range[1])
                ],
                comparison_metric,
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=chart_df["round"],
                    y=chart_df["mean"],
                    mode="lines+markers",
                    name="Mean",
                    line=dict(color="#60a5fa", width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=chart_df["round"],
                    y=chart_df["min"],
                    mode="lines",
                    name="Min",
                    line=dict(color="#f59e0b", width=1.5, dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=chart_df["round"],
                    y=chart_df["max"],
                    mode="lines",
                    name="Max",
                    line=dict(color="#10b981", width=1.5, dash="dash"),
                )
            )
            fig.update_layout(
                template="plotly_dark",
                height=340,
                margin=dict(l=10, r=10, t=40, b=10),
                title=f"Selected-range distribution for {comparison_metric.upper()}",
                xaxis_title="Round",
                yaxis_title=comparison_metric.upper(),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No statistics available for the selected range.")

    with tab2:
        st.subheader("Saved Comparison Charts")
        existing = [Path(p) for p in CHART_FILES if Path(p).exists()]
        if not existing:
            st.warning("No saved chart images found. Run main.py to generate plots.")
        else:
            cols = st.columns(2)
            for i, img_path in enumerate(existing):
                with cols[i % 2]:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("Algorithm Comparison Matrix")
            matrix_metric = st.selectbox("Matrix metric", ["f1", "f1_var", "cost", "auc"], index=0, key="matrix-metric")
            matrix = build_metric_matrix(ts_df, matrix_metric, selected_algorithms)
            st.dataframe(matrix.style.format("{:.4f}"), use_container_width=True)

    with tab3:
        st.subheader("Model Flow")
        st.markdown(
            """
1. Data loading and preprocessing
2. Client initialization and local model training
3. Candidate client subset generation by selector (Random/Greedy/PSO/GWO/Hybrid)
4. Server-side weighted aggregation of selected client updates
5. Global model evaluation (F1, AUC, fairness variance, communication cost)
6. Repeat for multiple communication rounds
"""
        )

        st.code(
            """
for round in rounds:
    for client in clients:
        client.train_local(global_weights)

    selection_mask, client_weights = selector.optimize_or_select()
    selected_updates = gather_updates(selection_mask)

    new_global = server.aggregate(selected_updates, client_weights)
    server.set_global_weights(new_global)

    metrics = server.evaluate()
""",
            language="python",
        )

        st.subheader("Architecture Snapshot")
        st.markdown(
            """
- Client Layer: heterogeneous IoT domains with private local datasets
- Optimization Layer: baseline selectors + metaheuristic selectors
- Hybrid Layer: PSO velocity memory + GWO alpha/beta/delta guidance
- Server Layer: weighted FedAvg-style aggregation and global validation
"""
        )
        st.markdown("<div class='small-note'>Tip: use the Hybrid vs Baselines tab to inspect how these architectural choices impact metric trajectories.</div>", unsafe_allow_html=True)

    with tab4:
        st.subheader("Detailed Round-wise Metrics")
        detail_cols = ["algorithm", "round", "f1", "f1_var", "cost", "auc", "f1_roll3", "f1_var_roll3", "cost_roll3", "auc_roll3"]
        filtered_df = ts_df[
            ts_df["algorithm"].isin(selected_algorithms)
            & (ts_df["round"] >= round_range[0])
            & (ts_df["round"] <= round_range[1])
        ]
        st.dataframe(filtered_df[detail_cols], use_container_width=True)

        st.download_button(
            "Download filtered metrics CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_metrics.csv",
            mime="text/csv",
        )

        st.subheader("Method Summary JSON")
        st.json(summary)


if __name__ == "__main__":
    main()
