# Hybrid PSO-GWO Federated IDS

This project implements a federated intrusion detection system for the ToN-IoT CSV files in this workspace. Each device CSV is treated as a federated client, a local MLP IDS is trained on each client, and a hybrid PSO-GWO selector chooses the participating clients every round.

## What is included

- Binary IDS preprocessing for the provided ToN-IoT CSV files.
- Federated training with FedAvg aggregation.
- Hybrid PSO-GWO client selection.
- Baselines: random, greedy, clustering-based, all-client FedAvg, plus PSO-only and GWO-only ablations.
- Streamlit dashboard for performance, fairness, convergence, communication, and ablation analysis.

## Run the experiment

```powershell
cd "d:\MHO implementation"
pip install -r requirements.txt
python hybrid_pso_gwo_ids.py --clients default --rounds 5 --strategies hybrid,random,greedy,clustered,all,pso,gwo
```

## Launch the dashboard

```powershell
streamlit run dashboard.py
```

The dashboard can also launch a fresh experiment from the sidebar and will load the latest saved run automatically.

## Notes

- The default configuration uses sampled client subsets so the experiment completes in a reasonable time.
- Increase `--max-rows-per-client`, `--rounds`, or the selector population/iterations for heavier runs.
- The dashboard reports the actual comparison results from the latest run artifacts; it does not hard-code a success claim.
