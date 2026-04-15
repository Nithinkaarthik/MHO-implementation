import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"


def list_run_dirs() -> List[Path]:
    if not OUTPUTS_DIR.exists():
        return []
    runs = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def load_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_round_details(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "round_metrics_all.csv"
    if path.exists():
        return pd.read_csv(path)

    csvs = list(run_dir.glob("round_metrics_*.csv"))
    chunks = []
    for c in csvs:
        if c.name == "round_metrics_all.csv":
            continue
        try:
            chunks.append(pd.read_csv(c))
        except Exception:
            continue
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def proof_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    if df.empty or "final_f1" not in df.columns:
        return pd.DataFrame()
    df = df.sort_values("final_f1", ascending=False).reset_index(drop=True)
    best_f1 = float(df.loc[0, "final_f1"])
    df["delta_to_best_f1"] = df["final_f1"] - best_f1
    return df


def claim_status(summary_df: pd.DataFrame) -> Dict[str, str]:
    if summary_df.empty:
        return {"status": "NO DATA", "message": "No summary file found in selected run."}

    if "strategy" not in summary_df.columns or "final_f1" not in summary_df.columns:
        return {"status": "NO DATA", "message": "Required columns missing in summary_comparison.csv."}

    ranking = summary_df.sort_values("final_f1", ascending=False).reset_index(drop=True)
    top_strategy = str(ranking.loc[0, "strategy"])

    if "mho" not in set(summary_df["strategy"]):
        return {"status": "NO MHO", "message": "Selected run has no mho entry."}

    mho_f1 = float(summary_df.loc[summary_df["strategy"] == "mho", "final_f1"].iloc[0])
    best_other = float(summary_df.loc[summary_df["strategy"] != "mho", "final_f1"].max())
    margin = mho_f1 - best_other

    if top_strategy == "mho":
        return {
            "status": "PASS",
            "message": f"Hybrid-MHO leads on final F1. Margin vs best baseline: {margin:.4f}",
        }

    return {
        "status": "FAIL",
        "message": f"Hybrid-MHO does not lead on final F1. Gap to leader: {abs(margin):.4f}",
    }


def get_recommended_proof_combo() -> pd.DataFrame:
    path = ROOT / "combo_search_hmho_seeded.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    needed = ["combo", "mho_f1", "best_other_f1", "margin", "best_strategy"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()
    return df.sort_values("margin", ascending=False)


def render_header() -> None:
    st.set_page_config(page_title="Hybrid-MHO FL-IDS Proof Dashboard", layout="wide")
    st.title("Hybrid-MHO FL-IDS Proof Dashboard")
    st.caption("Evidence dashboard for federated IoT intrusion detection with Hybrid-MHO client selection")


def render_method_explanation() -> None:
    st.subheader("Method: How Hybrid-MHO Works")
    st.markdown(
        """
Hybrid-MHO is implemented as the mho method and combines four ideas inside one optimization pipeline:

1. MHO client subset search with chaotic initialization, growth update, and reverse learning.
2. Greedy-seeded population so the optimizer starts near strong local-F1 client subsets.
3. Hybrid aggregation in each round: MHO weights blended with data-size priors.
4. Stability gate: candidate updates are validated and can be stabilized or rejected.

Optimization objective in proxy evaluation:

- maximize global F1
- maximize fairness (Jain index)
- minimize communication cost

Fed aggregation formula:

$$
\\theta_g = \\sum_{k \\in S} w_k \\cdot \\theta_k
$$
"""
    )


def render_proof_panel(summary_df: pd.DataFrame, run_dir: Path) -> None:
    st.subheader("Proof: Does Hybrid-MHO Outperform Baselines?")
    status = claim_status(summary_df)

    if status["status"] == "PASS":
        st.success(status["message"])
    elif status["status"] == "FAIL":
        st.error(status["message"])
    else:
        st.warning(status["message"])

    if summary_df.empty:
        return

    rank_df = proof_table(summary_df)
    st.dataframe(rank_df, use_container_width=True)

    if "strategy" in summary_df.columns and "final_f1" in summary_df.columns:
        fig = px.bar(
            summary_df.sort_values("final_f1", ascending=False),
            x="strategy",
            y="final_f1",
            color="strategy",
            title="Final F1 by Strategy",
            text="final_f1",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(3)
    if "mho" in set(summary_df["strategy"]):
        mho_row = summary_df[summary_df["strategy"] == "mho"].iloc[0]
        cols[0].metric("Hybrid-MHO Final F1", f"{mho_row['final_f1']:.4f}")
        cols[1].metric("Hybrid-MHO AUC", f"{mho_row['final_auc']:.4f}")
        cols[2].metric("Hybrid-MHO Avg Comm Cost", f"{mho_row['avg_comm_cost']:.3f}")

    st.caption(f"Evidence source: {run_dir}")


def render_round_trends(round_df: pd.DataFrame) -> None:
    st.subheader("Round-by-Round Trends")
    if round_df.empty:
        st.info("No round metrics found for selected run.")
        return

    needed = {"round", "strategy", "f1", "auc", "comm_cost"}
    if not needed.issubset(set(round_df.columns)):
        st.info("Round metrics file does not include required columns.")
        return

    fig_f1 = px.line(
        round_df,
        x="round",
        y="f1",
        color="strategy",
        markers=True,
        title="F1 Convergence by Round",
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    fig_auc = px.line(
        round_df,
        x="round",
        y="auc",
        color="strategy",
        markers=True,
        title="AUC by Round",
    )
    st.plotly_chart(fig_auc, use_container_width=True)

    fig_comm = px.bar(
        round_df,
        x="round",
        y="comm_cost",
        color="strategy",
        barmode="group",
        title="Communication Cost by Round",
    )
    st.plotly_chart(fig_comm, use_container_width=True)


def render_combo_proof() -> None:
    st.subheader("Proof Across 3-Client Combinations")
    combo_df = get_recommended_proof_combo()
    if combo_df.empty:
        st.info("combo_search_hmho_seeded.csv not found or invalid.")
        return

    wins = int((combo_df["best_strategy"] == "mho").sum())
    total = int(len(combo_df))
    win_rate = 100.0 * wins / max(total, 1)
    st.metric("Hybrid-MHO Best-Strategy Rate", f"{win_rate:.1f}%", f"{wins}/{total} combinations")

    top10 = combo_df.head(10).copy()
    st.dataframe(top10, use_container_width=True)

    fig = px.histogram(
        combo_df,
        x="margin",
        nbins=30,
        title="Distribution of Hybrid-MHO Margin vs Best Other Baseline",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_reproducibility(run_dir: Path) -> None:
    st.subheader("Reproducibility")
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            st.json(cfg)
        except Exception:
            st.warning("run_config.json exists but could not be parsed.")
    else:
        st.info("No run_config.json found for selected run.")


def main() -> None:
    render_header()

    runs = list_run_dirs()
    if not runs:
        st.error("No runs found under outputs/.")
        return

    run_names = [r.name for r in runs]
    default_idx = 0
    if "run_20260410_002855" in run_names:
        default_idx = run_names.index("run_20260410_002855")

    chosen = st.sidebar.selectbox("Select experiment run", run_names, index=default_idx)
    run_dir = OUTPUTS_DIR / chosen

    summary_df = load_summary(run_dir)
    round_df = load_round_details(run_dir)

    render_method_explanation()
    render_proof_panel(summary_df, run_dir)
    render_round_trends(round_df)
    render_combo_proof()
    render_reproducibility(run_dir)


if __name__ == "__main__":
    main()
