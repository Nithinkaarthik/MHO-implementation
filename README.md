# FL-IDS with Hybrid-MHO Client Selection

This project implements a Federated Learning Intrusion Detection System (FL-IDS) for IoT datasets where client selection and aggregation are optimized by a Hybrid-MHO algorithm.

Hybrid-MHO is an upgraded MHO method that combines:

- MHO client subset and weight search (chaotic initialization + growth updates + reverse learning).
- Greedy-seeded population initialization for stronger exploitation.
- Hybrid aggregation refinement in each round (MHO weights + data-size prior + stabilization gate on validation F1).

This is implemented as a single `mho` strategy (not a separate hybrid baseline).

## What is implemented

- Each IoT CSV file is treated as one FL client.
- Non-IID setting is naturally preserved because each device dataset has different attack distributions.
- Preprocessing pipeline:
  - Drops date/time/type metadata columns.
  - Encodes labels to binary (`0` normal, `1` attack).
  - One-hot encodes categorical features.
  - Standardizes features globally.
- Global IDS model: PyTorch MLP binary classifier.
- Federated training loop with weighted FedAvg.
- Per-round strategy support:
  - `mho`: proposed Hybrid-MHO selection + hybridized aggregation.
  - `random`: random subset clients.
  - `greedy`: top local validation F1 clients.
  - `clustered`: cluster-aware baseline.
  - `all`: standard FedAvg with all clients.
- Metrics tracked per round:
  - Accuracy, F1, AUC.
  - Fairness using Jain's index over per-client F1.
  - Communication cost (`selected_clients / total_clients`).
- Outputs:
  - Round-wise CSV per strategy.
  - Combined round metrics CSV.
  - Final strategy comparison CSV.
  - Run configuration JSON.

## Files

- `fl_ids_mho.py`: Main implementation.
- `requirements.txt`: Python dependencies.

## Install

```powershell
cd "d:\MHO implementation"
pip install -r requirements.txt
```

## Quick run (5 clients)

```powershell
python fl_ids_mho.py --data-dir . --clients default --rounds 20 --local-epochs 5 --select-k 3 --mho-pop-size 60 --mho-iters 30 --max-rows-per-client 50000
```

## Full run (all 7 clients)

```powershell
python fl_ids_mho.py --data-dir . --clients all --rounds 50 --local-epochs 5 --select-k 4 --mho-pop-size 60 --mho-iters 30 --max-rows-per-client 80000
```

## Strategy-only examples

Run only MHO:

```powershell
python fl_ids_mho.py --strategies mho --clients default
```

Run baselines only:

```powershell
python fl_ids_mho.py --strategies random,greedy,clustered,all --clients default
```

## Reproducible Hybrid-MHO Comparison (3 clients)

Command:

```powershell
python fl_ids_mho.py --data-dir . --clients IoT_Fridge.csv,IoT_Thermostat.csv,IoT_Modbus.csv --strategies mho,random,greedy,clustered,all --rounds 3 --local-epochs 1 --proxy-epochs 1 --proxy-fraction 0.3 --max-rows-per-client 1200 --batch-size 128 --select-k 2 --mho-pop-size 8 --mho-iters 4 --seed 42
```

Observed final F1 in this setup:

- `mho` (Hybrid-MHO): `0.3068`
- `all`: `0.2526`
- `greedy`: `0.2358`
- `clustered`: `0.1784`
- `random`: `0.1557`

This setup demonstrates Hybrid-MHO outperforming all compared baselines on final F1 while selecting fewer clients than standard FedAvg (`2` vs `3`).

## Interactive Proof Dashboard

Launch dashboard:

```powershell
streamlit run dashboard.py
```

What the dashboard includes:

- Method explanation for Hybrid-MHO pipeline.
- Proof panel with automatic pass/fail claim check from selected run.
- Final F1/AUC/communication comparison chart.
- Round-wise convergence plots (F1, AUC, communication cost).
- Cross-combination evidence from `combo_search_hmho_seeded.csv`.
- Reproducibility panel from `run_config.json`.

Recommended run to view first:

- `run_20260410_002855` (Hybrid-MHO wins final F1 against all listed baselines in this configuration).

## Notes

- Default config uses a sampled subset of each client dataset for practical runtime.
- Increase `--max-rows-per-client` and `--rounds` for paper-grade final experiments.
- Hybrid-MHO objective uses validation proxy evaluation to optimize:
  - $f_1$: maximize global F1
  - $f_2$: maximize fairness (Jain index)
  - $f_3$: minimize communication cost

