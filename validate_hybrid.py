import json
import numpy as np
from pathlib import Path

RESULTS_PATH = Path("results.json")
HYBRID_NAME = "Hybrid PSO-GWO"


def summarize_metrics(metrics):
    f1 = np.array(metrics["f1"], dtype=float)
    f1_var = np.array(metrics["f1_var"], dtype=float)
    cost = np.array(metrics["cost"], dtype=float)
    auc = np.array(metrics["auc"], dtype=float)

    # Composite utility: rewards quality, penalizes fairness variance and communication cost.
    utility = float(np.mean(f1 - 0.2 * f1_var - 0.1 * cost))

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


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("results.json not found. Run main.py first.")

    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        results = json.load(f)

    if HYBRID_NAME not in results:
        raise KeyError(f"{HYBRID_NAME} missing from results.json")

    summary = {name: summarize_metrics(m) for name, m in results.items()}
    hybrid = summary[HYBRID_NAME]

    baselines = {k: v for k, v in summary.items() if k != HYBRID_NAME}
    best_baseline_mean_f1 = max(v["mean_f1"] for v in baselines.values())
    best_baseline_best_f1 = max(v["best_f1"] for v in baselines.values())
    best_baseline_efficiency = max(v["efficiency"] for v in baselines.values())
    best_baseline_utility = max(v["utility"] for v in baselines.values())

    print("=== Summary ===")
    for name, vals in summary.items():
        print(
            f"{name:15s} | mean_f1={vals['mean_f1']:.4f} | final_f1={vals['final_f1']:.4f} | "
            f"best_f1={vals['best_f1']:.4f} | "
            f"mean_cost={vals['mean_cost']:.4f} | mean_var={vals['mean_var']:.6f} | "
            f"eff={vals['efficiency']:.4f} | utility={vals['utility']:.4f}"
        )

    print("\n=== Hybrid Validation ===")
    checks = {
        "mean_f1_beats_all_baselines": hybrid["mean_f1"] > best_baseline_mean_f1,
        "best_round_f1_beats_all_baselines": hybrid["best_f1"] > best_baseline_best_f1,
        "selection_efficiency_beats_all_baselines": hybrid["efficiency"] > best_baseline_efficiency,
        "overall_utility_beats_all_baselines": hybrid["utility"] > best_baseline_utility,
    }

    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")

    passed = all(checks.values())
    print("\nRESULT:", "PASS" if passed else "FAIL")

    # Non-zero exit code makes this usable in CI-style checks.
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
