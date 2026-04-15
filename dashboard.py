import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    "PSO": "#f97316",
    "GWO": "#14b8a6",
    "Hybrid PSO-GWO": "#16a34a",
}


def inject_theme():
    st.markdown(
        """
<style>
:root {
    --bg: #f5f8ff;
    --panel: #ffffff;
    --ink: #101828;
    --muted: #475467;
    --line: #d0d5dd;
    --accent: #0b7285;
    --accent-soft: #e6f6f9;
    --good: #15803d;
    --good-bg: #e8f7ec;
    --bad: #b42318;
    --bad-bg: #fef3f2;
    --hero-1: #0b7285;
    --hero-2: #2563eb;
}

.stApp {
    background:
      radial-gradient(1200px 380px at 10% -20%, #e0f2fe 0%, transparent 60%),
      radial-gradient(900px 300px at 95% 0%, #dcfce7 0%, transparent 55%),
      var(--bg);
}

.hero {
    padding: 1.2rem 1.4rem;
    border-radius: 16px;
    background: linear-gradient(120deg, var(--hero-1), var(--hero-2));
    box-shadow: 0 10px 30px rgba(11, 114, 133, 0.28);
    color: white;
    margin-bottom: 1rem;
}

.hero h1 {
    margin: 0;
    font-size: 1.9rem;
    letter-spacing: 0.2px;
}

.hero p {
    margin: 0.35rem 0 0 0;
    opacity: 0.94;
    font-size: 1rem;
}

.section-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
}

.kpi {
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
}

.kpi .label {
    color: var(--muted);
    font-size: 0.82rem;
    margin-bottom: 0.1rem;
}

.kpi .value {
    color: var(--ink);
    font-size: 1.22rem;
    font-weight: 700;
}

.badge-pass {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    color: var(--good);
    background: var(--good-bg);
    border: 1px solid #b7e4c7;
    font-weight: 700;
    font-size: 0.78rem;
}

.badge-fail {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    color: var(--bad);
    background: var(--bad-bg);
    border: 1px solid #fecdd3;
    font-weight: 700;
    font-size: 0.78rem;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.45rem 0.6rem;
}

.small-note {
    color: var(--muted);
    font-size: 0.86rem;
}

@media (max-width: 900px) {
    .hero h1 { font-size: 1.4rem; }
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

    utility = float(np.mean(f1 - fairness_weight * f1_var - cost_weight * cost))

    return {
        "mean_f1": float(np.mean(f1)),
        "final_f1": float(f1[-1]),
        "best_f1": float(np.max(f1)),
        "mean_auc": float(np.mean(auc)),
        "mean_var": float(np.mean(f1_var)),
        "mean_cost": float(np.mean(cost)),
        "efficiency": float(np.mean(f1 / (cost + 1e-12))),
        "utility": utility,
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
    return pd.DataFrame(rows)


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


def render_metric_plot(df, metric, selected_algorithms):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    sub = df[df["algorithm"].isin(selected_algorithms)]
    for algo in selected_algorithms:
        s = sub[sub["algorithm"] == algo]
        if not s.empty:
            color = ALGO_COLORS.get(algo, "#334155")
            width = 2.8 if algo == HYBRID_NAME else 2.0
            alpha = 1.0 if algo == HYBRID_NAME else 0.88
            ax.plot(
                s["round"],
                s[metric],
                marker="o",
                linewidth=width,
                color=color,
                alpha=alpha,
                label=algo,
            )
    ax.set_xlabel("Round")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs Round")
    ax.set_facecolor("#fbfcfe")
    ax.grid(alpha=0.24, linestyle="--")
    for spine in ax.spines.values():
        spine.set_alpha(0.35)
    ax.legend(loc="best")
    fig.patch.set_alpha(0)
    plt.tight_layout()
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Hybrid FL Dashboard", layout="wide")
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
        st.markdown("<hr style='border: 0; border-top: 1px solid #d0d5dd;'>", unsafe_allow_html=True)
        st.markdown(f"<b>overall_result</b><br>{status_badge(overall)}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Hybrid vs Baselines", "Comparison Charts", "Model Flow", "Raw Metrics"]
    )

    with tab1:
        st.subheader("Why Hybrid Beats Baselines")
        styled = (
            table.style.format("{:.4f}")
            .background_gradient(cmap="Blues", subset=["mean_f1", "best_f1", "efficiency", "utility"])
            .background_gradient(cmap="RdYlGn_r", subset=["mean_cost", "mean_var"])
        )
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
        all_algorithms = list(results.keys())
        default_algorithms = [HYBRID_NAME] + [a for a in all_algorithms if a != HYBRID_NAME][:3]
        selected_algorithms = st.multiselect(
            "Algorithms to plot", all_algorithms, default=default_algorithms
        )
        metric = st.selectbox("Metric", ["f1", "f1_var", "cost", "auc"], index=0)

        if selected_algorithms:
            render_metric_plot(ts_df, metric, selected_algorithms)
        else:
            st.warning("Select at least one algorithm.")

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
        st.dataframe(ts_df, use_container_width=True)

        st.subheader("Method Summary JSON")
        st.json(summary)


if __name__ == "__main__":
    main()
