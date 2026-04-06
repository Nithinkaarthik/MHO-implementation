import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="MHO-FL IoT IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM THEME & STYLING ====================
# Electricity-inspired professional theme with dark background
theme_css = """
<style>
    :root {
        --primary-color: #00D9FF;
        --secondary-color: #FFD700;
        --accent-color: #1A1A2E;
        --success-color: #00FF41;
        --warning-color: #FF6B6B;
        --dark-bg: #0F0F1E;
        --card-bg: #1A1A2E;
    }
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0F0F1E 0%, #16213E 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%);
        border-right: 2px solid rgba(0, 217, 255, 0.3);
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #00D9FF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
    }
    
    /* Text styling */
    p, span, li, div {
        color: #E0E0E0 !important;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: 10px;
        padding: 20px !important;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.1) !important;
    }
    
    /* Info boxes */
    [data-testid="stAlert"] {
        background: rgba(0, 217, 255, 0.1) !important;
        border-left: 4px solid #00D9FF !important;
        border-radius: 5px !important;
    }
    
    /* Buttons */
    button {
        background: linear-gradient(135deg, #00D9FF 0%, #00A8CC 100%) !important;
        color: #0F0F1E !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    button:hover {
        box-shadow: 0 0 25px rgba(0, 217, 255, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Select boxes and inputs */
    [data-testid="stSelectbox"] div, [data-testid="stCheckbox"] div {
        color: #E0E0E0 !important;
    }
    
    /* Tabs */
    [data-testid="stTabs"] [role="tab"] {
        color: #B0B0B0 !important;
        border-bottom: 2px solid transparent !important;
    }
    
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #00D9FF !important;
        border-bottom: 3px solid #00D9FF !important;
        box-shadow: 0 2px 10px rgba(0, 217, 255, 0.2) !important;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        background: rgba(26, 26, 46, 0.8) !important;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: rgba(26, 26, 46, 0.6) !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    
    /* Caption text */
    .small, caption {
        color: #888888 !important;
    }
    
    /* Success/Error messages */
    .success { color: #00FF41 !important; }
    .error { color: #FF6B6B !important; }
    .warning { color: #FFD700 !important; }
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)


def discover_output_dirs(base_dir: Path) -> list[Path]:
    candidates = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("outputs_garage"):
            summary = item / "summary_metrics.json"
            if summary.exists():
                candidates.append(item)
    return sorted(candidates, key=lambda p: p.name)


def load_summary(summary_path: Path) -> dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summary_to_dataframe(summary: dict) -> pd.DataFrame:
    rows = []
    for method, payload in summary.items():
        test = payload.get("final_test", {})
        fair = payload.get("final_fairness", {})
        rows.append(
            {
                "method": method,
                "accuracy": test.get("accuracy", 0.0),
                "precision": test.get("precision", 0.0),
                "recall": test.get("recall", 0.0),
                "f1": test.get("f1", 0.0),
                "auc": test.get("auc", 0.0),
                "client_f1_var": fair.get("client_f1_var", 0.0),
                "communication_MB": payload.get("communication_MB", 0.0),
                "convergence_round": payload.get("convergence_round_95pct_peak_f1", 0),
            }
        )

    return pd.DataFrame(rows)


def get_dataset_profile(csv_path: Path) -> tuple[int, pd.DataFrame]:
    if not csv_path.exists():
        return 0, pd.DataFrame(columns=["type", "count"])

    df = pd.read_csv(csv_path, usecols=["type"])
    counts = df["type"].astype(str).str.strip().value_counts().rename_axis("type").reset_index(name="count")
    return len(df), counts


def method_label(method_name: str) -> str:
    mapping = {
        "all_clients_fedavg": "FedAvg (All Clients)",
        "random": "Random Selection",
        "greedy": "Greedy Selection",
        "cluster": "Clustering Selection",
        "mho": "MHO Selection",
    }
    return mapping.get(method_name, method_name)


def display_project_overview() -> None:
    # Professional title with branding
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="
                font-size: 2.8em;
                background: linear-gradient(90deg, #00D9FF, #FFD700);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            ">⚡ Metaheuristic-Optimized Federated IDS</h1>
            <p style="font-size: 1.1em; color: #00D9FF;">Intelligent Heterogeneous IoT Intrusion Detection</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <p style="text-align: center; color: #B0B0B0; margin: 20px 0;">
        This dashboard visualizes a federated intrusion detection framework on the <b>ToN-IoT Garage Door</b> dataset.
        <br>
        <b>Modified Hippopotamus Optimization (MHO)</b> intelligently selects clients and determines aggregation weights.
        </p>
    """, unsafe_allow_html=True)

    # Objectives with styled cards
    col1, col2, col3 = st.columns(3)
    
    objective_card = """
    <div style="
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%);
        border: 2px solid {color};
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 0 15px {shadow};
    ">
        <h3 style="color: {color}; margin-top: 0;">⚙️ {title}</h3>
        <p style="color: #E0E0E0;">{content}</p>
    </div>
    """
    
    with col1:
        st.markdown(objective_card.format(
            color="#00D9FF",
            shadow="rgba(0, 217, 255, 0.1)",
            title="Objective 1",
            content="Maximize IDS Quality: Achieve optimal F1 and AUC scores for precise threat detection."
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(objective_card.format(
            color="#00FF41",
            shadow="rgba(0, 255, 65, 0.1)",
            title="Objective 2",
            content="Improve Fairness: Balance performance across heterogeneous IoT clients."
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(objective_card.format(
            color="#FFD700",
            shadow="rgba(255, 215, 0, 0.1)",
            title="Objective 3",
            content="Minimize Communication: Reduce bandwidth overhead in federated rounds."
        ), unsafe_allow_html=True)

    with st.expander("🔧 Federated Learning + MHO Workflow", expanded=True):
        st.markdown("""
            <ol style="color: #E0E0E0; line-height: 1.8;">
                <li><strong style="color: #00D9FF;">Data Preprocessing:</strong> Garage Door dataset partitioned into non-IID clients.</li>
                <li><strong style="color: #00D9FF;">Model Initialization:</strong> Global IDS model initialized at server.</li>
                <li><strong style="color: #00D9FF;">MHO Optimization:</strong> Each round explores candidate client subsets and aggregation weights.</li>
                <li><strong style="color: #00D9FF;">Local Training:</strong> Selected clients train independently on local data.</li>
                <li><strong style="color: #00D9FF;">Weighted Aggregation:</strong> Server performs intelligent FedAvg with MHO-determined weights.</li>
                <li><strong style="color: #00D9FF;">Evaluation & Iteration:</strong> Global and per-client metrics evaluated; repeat until convergence.</li>
            </ol>
        """, unsafe_allow_html=True)


base_dir = Path(__file__).resolve().parent
output_dirs = discover_output_dirs(base_dir)

# Enhanced sidebar
st.sidebar.markdown("""
    <div style="text-align: center; padding: 15px 0;">
        <h2 style="color: #00D9FF; margin: 0;">⚡ Controls</h2>
        <p style="color: #888; font-size: 0.8em;">Configure Dashboard</p>
    </div>
""", unsafe_allow_html=True)

if not output_dirs:
    st.sidebar.error("❌ No outputs_garage* folder with summary_metrics.json found.")
    st.stop()

selected_dir = st.sidebar.selectbox(
    "📂 Select Experiment Output",
    options=output_dirs,
    format_func=lambda p: p.name,
)

summary_path = selected_dir / "summary_metrics.json"
summary = load_summary(summary_path)
df = summary_to_dataframe(summary)
df["Method"] = df["method"].apply(method_label)

metric_focus = st.sidebar.selectbox(
    "📊 Primary Performance Metric",
    options=["f1", "auc", "accuracy", "precision", "recall"],
    index=0,
)

show_raw = st.sidebar.checkbox("🔍 Show Raw JSON", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="
        background: rgba(0, 217, 255, 0.1);
        border-left: 3px solid #00D9FF;
        padding: 12px;
        border-radius: 5px;
        margin-top: 20px;
    ">
        <p style="margin: 0; color: #888; font-size: 0.75em; font-weight: bold;">DATA SOURCE</p>
        <p style="margin: 5px 0 0 0; color: #00D9FF; font-weight: 600;">
            📡 IoT_Garage_Door.csv
        </p>
    </div>
""", unsafe_allow_html=True)

# Main layout tabs
project_tab, data_tab, results_tab, tradeoff_tab, insights_tab = st.tabs(
    [
        "Project Overview",
        "Dataset Profile",
        "Method Comparison",
        "Fairness vs Communication",
        "Interpretation",
    ]
)

with project_tab:
    display_project_overview()

    st.markdown("---")
    
    col_obj1, col_obj2 = st.columns([3, 1])
    with col_obj1:
        st.subheader("🎯 Multi-Objective Optimization Formula")
        st.markdown("""
            <p style="color: #E0E0E0; font-size: 1.05em; line-height: 1.6;">
            The MHO algorithm minimizes a weighted combination of three objectives:
            </p>
        """, unsafe_allow_html=True)
        st.latex(
            r"J = \alpha (1 - F1_{global}) + \beta\,\text{Var}(F1_{clients}) + \gamma\,\text{CommCost}"
        )
        st.markdown("""
            <div style="background: rgba(0, 217, 255, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid #00D9FF;">
                <p style="color: #E0E0E0; margin: 0; font-size: 0.9em;">
                    <strong>Where:</strong> α controls IDS quality weight • β controls fairness weight • γ controls communication weight
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_obj2:
        st.metric("🔋 Algorithm", "MHO", delta="Modified Hippo")
        st.metric("🎯 Objectives", "3", delta="Multi-criteria")

with data_tab:
    st.subheader("📊 Garage Door Dataset Snapshot")
    rows, type_counts = get_dataset_profile(base_dir / "IoT_Garage_Door.csv")

    # Enhanced metrics display
    k1, k2, k3, k4 = st.columns(4)
    
    metric_style = """
    <div style="
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.15) 0%, rgba(255, 215, 0, 0.08) 100%);
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.1);
    ">
        <p style="color: #888; font-size: 0.85em; margin: 0; text-transform: uppercase;">
            {label}
        </p>
        <h3 style="color: #00D9FF; margin: 10px 0 0 0; font-size: 2em;">
            {value}
        </h3>
    </div>
    """
    
    with k1:
        st.markdown(metric_style.format(label="Total Records", value=f"{rows:,}"), unsafe_allow_html=True)
    with k2:
        st.markdown(metric_style.format(label="Attack Types", value=int(type_counts.shape[0])), unsafe_allow_html=True)
    with k3:
        st.markdown(metric_style.format(label="Dataset", value="Garage Door"), unsafe_allow_html=True)
    with k4:
        st.markdown(metric_style.format(label="Status", value="✓ Ready"), unsafe_allow_html=True)

    if not type_counts.empty:
        st.markdown("#### Traffic Type Distribution")
        fig_types = px.bar(
            type_counts,
            x="type",
            y="count",
            color="count",
            color_continuous_scale="Viridis",
            title="",
        )
        fig_types.update_layout(
            xaxis_title="Traffic Type",
            yaxis_title="Count",
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E0E0"),
            hovermode="closest",
        )
        fig_types.update_traces(marker_line_color="rgba(0, 217, 255, 0.3)", marker_line_width=1)
        st.plotly_chart(fig_types, use_container_width=True)

        st.markdown("#### Detailed Type Breakdown")
        st.dataframe(
            type_counts.style.format({"count": "{:,.0f}"}),
            use_container_width=True
        )

with results_tab:
    st.subheader("🏆 Final Performance Across Client Selection Strategies")

    metrics_cols = ["accuracy", "precision", "recall", "f1", "auc"]
    df_long = df.melt(
        id_vars=["Method"],
        value_vars=metrics_cols,
        var_name="metric",
        value_name="value",
    )

    fig_perf = px.bar(
        df_long,
        x="Method",
        y="value",
        color="metric",
        barmode="group",
        title="Classification Metrics Comparison",
        color_discrete_sequence=["#00D9FF", "#00FF41", "#FFD700", "#FF6B6B", "#00A8CC"]
    )
    fig_perf.update_layout(
        yaxis_title="Score",
        xaxis_title="Client Selection Method",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
        hovermode="closest",
        height=450,
    )
    fig_perf.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,217,255,0.1)")
    fig_perf.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,217,255,0.1)")
    st.plotly_chart(fig_perf, use_container_width=True)

    rank_df = df[["Method", metric_focus]].sort_values(metric_focus, ascending=False).reset_index(drop=True)
    rank_df.index = rank_df.index + 1
    rank_df.columns = ["Method", metric_focus.upper()]
    
    st.markdown(f"#### 🥇 Ranking by {metric_focus.upper()}")
    
    # Create styled dataframe
    styled_rank = rank_df.style.format({metric_focus.upper(): "{:.4f}"})
    styled_rank = styled_rank.background_gradient(
        subset=[metric_focus.upper()],
        cmap="viridis",
        vmin=0,
        vmax=1
    )
    st.dataframe(styled_rank, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 Training Dynamics")
    
    cimg1, cimg2 = st.columns(2)
    conv_path = selected_dir / "convergence_f1.png"
    comm_path = selected_dir / "communication_overhead.png"

    with cimg1:
        st.markdown("**Convergence Profile**")
        if conv_path.exists():
            st.image(str(conv_path), use_container_width=True)
        else:
            st.info("⏳ convergence_f1.png not found in selected output folder.")

    with cimg2:
        st.markdown("**Communication Efficiency**")
        if comm_path.exists():
            st.image(str(comm_path), use_container_width=True)
        else:
            st.info("⏳ communication_overhead.png not found in selected output folder.")

with tradeoff_tab:
    st.subheader("⚖️ Multi-Objective Trade-off Analysis")

    st.markdown("""
        <div style="background: rgba(0, 217, 255, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid #FFD700; margin-bottom: 20px;">
            <p style="color: #E0E0E0; margin: 0; font-size: 0.9em;">
                <strong>Interpretation:</strong> Methods in the <strong>lower-left corner</strong> are optimal 
                (low communication cost + balanced fairness). Size represents the primary metric ({metric}).
            </p>
        </div>
    """.format(metric=metric_focus.upper()), unsafe_allow_html=True)

    fig_trade = px.scatter(
        df,
        x="communication_MB",
        y="client_f1_var",
        color="Method",
        size=metric_focus,
        hover_data={
            "f1": ":.4f",
            "auc": ":.4f",
            "accuracy": ":.4f",
            "convergence_round": True,
            "communication_MB": ":.3f",
            "client_f1_var": ":.5f",
        },
        title="",
        color_discrete_sequence=["#00D9FF", "#00FF41", "#FFD700", "#FF6B6B", "#00A8CC"],
    )
    fig_trade.update_layout(
        xaxis_title="💾 Communication Cost (MB)",
        yaxis_title="📊 Per-Client F1 Variance (Fairness)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
        hovermode="closest",
        height=500,
    )
    fig_trade.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,217,255,0.1)")
    fig_trade.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,217,255,0.1)")
    st.plotly_chart(fig_trade, use_container_width=True)

    st.markdown("#### ⚡ Convergence Speed Comparison")
    
    fig_conv = px.bar(
        df.sort_values("convergence_round"),
        x="Method",
        y="convergence_round",
        color="Method",
        title="",
        color_discrete_sequence=["#00D9FF", "#00FF41", "#FFD700", "#FF6B6B", "#00A8CC"],
    )
    fig_conv.update_layout(
        yaxis_title="Rounds to 95% Peak F1",
        xaxis_title="Client Selection Method",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
        hovermode="closest",
        showlegend=False,
        height=400,
    )
    fig_conv.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,217,255,0.1)")
    fig_conv.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,217,255,0.1)")
    st.plotly_chart(fig_conv, use_container_width=True)

with insights_tab:
    st.subheader("💡 Key Insights & Interpretation Guide")
    
    # Interpretation guide with styled cards
    guide_col1, guide_col2 = st.columns(2)
    
    insight_card = """
    <div style="
        background: rgba({r}, {g}, {b}, 0.1);
        border-left: 4px solid rgb({r}, {g}, {b});
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    ">
        <p style="color: #E0E0E0; margin: 0; line-height: 1.6;">
            <strong style="color: rgb({r}, {g}, {b});">{title}</strong><br>
            {content}
        </p>
    </div>
    """
    
    with guide_col1:
        st.markdown(insight_card.format(
            r=0, g=217, b=255,
            title="🎯 F1/AUC Score",
            content="Higher values indicate stronger intrusion detection quality and model precision."
        ), unsafe_allow_html=True)
        
        st.markdown(insight_card.format(
            r=0, g=255, b=65,
            title="⚖️ Client F1 Variance",
            content="Lower variance means fairer performance across heterogeneous IoT devices."
        ), unsafe_allow_html=True)
    
    with guide_col2:
        st.markdown(insight_card.format(
            r=255, g=215, b=0,
            title="💾 Communication Cost",
            content="Lower MB usage reduces bandwidth and energy consumption in federated rounds."
        ), unsafe_allow_html=True)
        
        st.markdown(insight_card.format(
            r=0, g=168, b=204,
            title="⚡ Convergence Speed",
            content="Fewer rounds to peak F1 means faster model stabilization and reduced training time."
        ), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🏅 Performance Summary")

    best_primary = df.sort_values(metric_focus, ascending=False).iloc[0]
    best_fair = df.sort_values("client_f1_var", ascending=True).iloc[0]
    best_comm = df.sort_values("communication_MB", ascending=True).iloc[0]

    col_best1, col_best2, col_best3 = st.columns(3)
    
    best_card = """
    <div style="
        background: linear-gradient(135deg, rgba({r}, {g}, {b}, 0.15) 0%, rgba(255, 215, 0, 0.05) 100%);
        border: 2px solid rgb({r}, {g}, {b});
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 10px rgba({r}, {g}, {b}, 0.2);
    ">
        <p style="color: #888; font-size: 0.8em; margin: 0; text-transform: uppercase; font-weight: bold;">
            {label}
        </p>
        <h3 style="color: rgb({r}, {g}, {b}); margin: 8px 0; font-size: 1.3em;">
            {method}
        </h3>
        <p style="color: #E0E0E0; margin: 8px 0 0 0; font-size: 1.1em; font-weight: bold;">
            {value}
        </p>
    </div>
    """
    
    with col_best1:
        st.markdown(best_card.format(
            r=0, g=217, b=255,
            label=f"Best {metric_focus.upper()}",
            method=best_primary['Method'],
            value=f"{best_primary[metric_focus]:.4f}"
        ), unsafe_allow_html=True)
    
    with col_best2:
        st.markdown(best_card.format(
            r=0, g=255, b=65,
            label="Best Fairness",
            method=best_fair['Method'],
            value=f"Var: {best_fair['client_f1_var']:.5f}"
        ), unsafe_allow_html=True)
    
    with col_best3:
        st.markdown(best_card.format(
            r=255, g=215, b=0,
            label="Lowest Communication",
            method=best_comm['Method'],
            value=f"{best_comm['communication_MB']:.3f} MB"
        ), unsafe_allow_html=True)

    if show_raw:
        st.markdown("---")
        st.markdown("#### 🔧 Raw Configuration Data (JSON)")
        st.json(summary)

st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; padding: 15px; color: #666; font-size: 0.85em;">
        <p>📊 Dashboard loaded from: <strong>{selected_dir.name}</strong><br>
        🛡️ MHO-Federated IDS | <strong style="color: #00D9FF;">Professional Analytics Dashboard</strong></p>
    </div>
""", unsafe_allow_html=True)
