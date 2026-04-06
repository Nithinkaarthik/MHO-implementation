# MHO-Optimized Federated IDS (Garage Door Only)

This implementation runs a full federated intrusion detection experiment on the **ToN-IoT Garage Door** CSV file (`IoT_Garage_Door.csv`) with heterogeneous non-IID clients.

## What is implemented

- Data preprocessing for Garage Door CSV:
  - cleans string fields,
  - creates time-derived features from `date` + `time`,
  - removes raw identifiers/timestamps from model input,
  - one-hot encodes categorical features,
  - standardizes features,
  - train/test split.
- Non-IID client simulation using attack-type skew with a Dirichlet partition.
- Local IDS model: PyTorch MLP classifier.
- Federated training with weighted FedAvg.
- MHO-based client selection + aggregation weights (multi-objective):
  - maximize detection (proxy F1),
  - maximize fairness (minimize variance of client F1),
  - minimize communication cost.
- Baseline methods:
  - all-clients FedAvg,
  - random client selection,
  - greedy selection,
  - clustering-based selection.
- Evaluation outputs:
  - accuracy, precision, recall, F1, AUC,
  - convergence speed,
  - communication overhead,
  - client fairness/per-client variance.

## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python mho_federated_garage.py --data IoT_Garage_Door.csv --rounds 10 --sample-fraction 0.10
```

## Useful options

- `--num-clients 8`
- `--num-selected-clients 4`
- `--local-epochs 1`
- `--batch-size 256`
- `--dirichlet-alpha 0.35`
- `--output-dir outputs_garage`

## Outputs

The script writes:

- `outputs_garage/summary_metrics.json`
- `outputs_garage/convergence_f1.png`
- `outputs_garage/communication_overhead.png`

## Streamlit Dashboard

An interactive project dashboard is included in `dashboard_streamlit.py`.

Run:

```powershell
streamlit run dashboard_streamlit.py
```

What it shows:

- project methodology and MHO objective,
- Garage Door dataset profile,
- comparison of MHO vs baselines,
- fairness vs communication trade-off,
- convergence and communication plots from experiment outputs.

## Note

Default arguments use `sample_fraction=0.10` for faster experimentation. Set `--sample-fraction 1.0` for full-data runs.
