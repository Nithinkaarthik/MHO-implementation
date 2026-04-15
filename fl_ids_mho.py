import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_CLIENT_FILES = [
    "IoT_Fridge.csv",
    "IoT_Garage_Door.csv",
    "IoT_Thermostat.csv",
    "IoT_Motion_Light.csv",
    "IoT_Weather.csv",
]

ALL_CLIENT_FILES = [
    "IoT_Fridge.csv",
    "IoT_Garage_Door.csv",
    "IoT_Thermostat.csv",
    "IoT_Motion_Light.csv",
    "IoT_Weather.csv",
    "IoT_Modbus.csv",
    "IoT_GPS_Tracker.csv",
]


@dataclass
class ClientData:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.y_train)


class IDSMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_with_fallback_stratify(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique, counts = np.unique(y, return_counts=True)
    stratify_ok = len(unique) > 1 and np.min(counts) >= 2
    stratify = y if stratify_ok else None
    return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=stratify)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in ["date", "time", "ts", "timestamp", "type"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip().str.lower()

    return df


def parse_labels(df: pd.DataFrame) -> np.ndarray:
    if "label" in df.columns:
        series = df["label"]
        if series.dtype == "object":
            s = series.astype(str).str.strip().str.lower()
            y = (s != "0").astype(np.int64)
        else:
            y = (pd.to_numeric(series, errors="coerce").fillna(0).astype(int) != 0).astype(np.int64)
        return y.to_numpy()

    if "type" in df.columns:
        s = df["type"].astype(str).str.strip().str.lower()
        return (s != "normal").astype(np.int64).to_numpy()

    raise ValueError("Could not find label/type column for target encoding.")


def sample_client_df(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df

    if "label" in df.columns:
        strat_col = df["label"]
    elif "type" in df.columns:
        strat_col = df["type"]
    else:
        strat_col = None

    if strat_col is not None:
        vc = pd.Series(strat_col).value_counts()
        stratify = strat_col if len(vc) > 1 and vc.min() >= 2 else None
    else:
        stratify = None

    sampled, _ = train_test_split(
        df,
        train_size=max_rows,
        random_state=seed,
        stratify=stratify,
    )
    return sampled.reset_index(drop=True)


def load_and_preprocess_clients(
    data_dir: str,
    client_files: Sequence[str],
    max_rows_per_client: int,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[List[ClientData], int, np.ndarray, np.ndarray]:
    raw_client_frames: List[pd.DataFrame] = []
    y_by_client: Dict[str, np.ndarray] = {}

    for fname in client_files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing client CSV: {path}")

        df = pd.read_csv(path, low_memory=False)
        df = sample_client_df(df, max_rows=max_rows_per_client, seed=seed)
        y = parse_labels(df)
        x_df = clean_dataframe(df.drop(columns=[c for c in ["label"] if c in df.columns]))
        x_df["__client_name"] = fname
        raw_client_frames.append(x_df)
        y_by_client[fname] = y

    merged = pd.concat(raw_client_frames, axis=0, ignore_index=True)

    client_col = merged["__client_name"].copy()
    feat_df = merged.drop(columns=["__client_name"])

    feat_df = feat_df.fillna(0)
    feat_df = pd.get_dummies(feat_df, drop_first=False)

    scaler = StandardScaler()
    x_all = scaler.fit_transform(feat_df.values.astype(np.float32)).astype(np.float32)

    x_by_client: Dict[str, np.ndarray] = {}
    start = 0
    for fname in client_files:
        n = len(y_by_client[fname])
        x_by_client[fname] = x_all[start : start + n]
        start += n

    clients: List[ClientData] = []
    global_x_test: List[np.ndarray] = []
    global_y_test: List[np.ndarray] = []

    for idx, fname in enumerate(client_files):
        x = x_by_client[fname]
        y = y_by_client[fname]

        x_train, x_tmp, y_train, y_tmp = split_with_fallback_stratify(x, y, test_size=val_size + test_size, seed=seed + idx)

        rel_test_size = test_size / (val_size + test_size)
        x_val, x_test, y_val, y_test = split_with_fallback_stratify(
            x_tmp,
            y_tmp,
            test_size=rel_test_size,
            seed=seed + idx + 1000,
        )

        clients.append(
            ClientData(
                name=fname,
                x_train=x_train,
                y_train=y_train.astype(np.int64),
                x_val=x_val,
                y_val=y_val.astype(np.int64),
                x_test=x_test,
                y_test=y_test.astype(np.int64),
            )
        )

        global_x_test.append(x_test)
        global_y_test.append(y_test)

    x_global_test = np.concatenate(global_x_test, axis=0)
    y_global_test = np.concatenate(global_y_test, axis=0)

    return clients, x_all.shape[1], x_global_test, y_global_test


def build_model(input_dim: int, device: torch.device) -> IDSMLP:
    model = IDSMLP(input_dim=input_dim)
    return model.to(device)


def to_tensor_dataset(x: np.ndarray, y: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tx = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    ty = torch.from_numpy(y).to(device=device, dtype=torch.float32)
    return tx, ty


def train_local_model(
    global_state: Dict[str, torch.Tensor],
    input_dim: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model = build_model(input_dim=input_dim, device=device)
    model.load_state_dict(global_state)
    model.train()

    tx, ty = to_tensor_dataset(x_train, y_train, device)

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    if pos > 0:
        pos_weight = torch.tensor([neg / pos], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(y_train)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            logits = model(tx[idx])
            loss = criterion(logits, ty[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def model_predict_proba(
    model_state: Dict[str, torch.Tensor],
    input_dim: int,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = build_model(input_dim=input_dim, device=device)
    model.load_state_dict(model_state)
    model.eval()

    with torch.no_grad():
        tx = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        logits = model(tx)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


def evaluate_state(
    model_state: Dict[str, torch.Tensor],
    input_dim: int,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    probs = model_predict_proba(model_state, input_dim=input_dim, x=x, device=device)
    preds = (probs >= 0.5).astype(np.int64)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)

    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "auc": float(auc),
    }


def jains_fairness(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    numerator = np.sum(arr) ** 2
    denominator = len(arr) * np.sum(arr ** 2)
    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def weighted_fedavg(states: List[Dict[str, torch.Tensor]], weights: np.ndarray) -> Dict[str, torch.Tensor]:
    if len(states) == 0:
        raise ValueError("No client states provided for aggregation.")

    w = np.asarray(weights, dtype=np.float64)
    w = w / (np.sum(w) + 1e-12)

    keys = states[0].keys()
    agg: Dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = torch.stack([s[k].float() for s in states], dim=0)
        weight_tensor = torch.tensor(w, dtype=torch.float32).view(-1, *([1] * (stacked.dim() - 1)))
        agg[k] = torch.sum(stacked * weight_tensor, dim=0)

    return agg


def blend_model_states(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended: Dict[str, torch.Tensor] = {}
    for k in state_a.keys():
        blended[k] = (1.0 - alpha) * state_a[k].float() + alpha * state_b[k].float()
    return blended


def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def make_proxy_updates(
    global_state: Dict[str, torch.Tensor],
    clients: Sequence[ClientData],
    input_dim: int,
    proxy_epochs: int,
    proxy_fraction: float,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> List[Dict[str, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    proxy_states: List[Dict[str, torch.Tensor]] = []

    for i, c in enumerate(clients):
        n = len(c.y_train)
        sub_n = max(256, int(n * proxy_fraction))
        sub_n = min(sub_n, n)
        idx = rng.choice(n, size=sub_n, replace=False)

        state = train_local_model(
            global_state=global_state,
            input_dim=input_dim,
            x_train=c.x_train[idx],
            y_train=c.y_train[idx],
            epochs=proxy_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        proxy_states.append(state)

    return proxy_states


def dominates(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
    # Objectives are (f1, fairness, comm_cost) where first two are maximize, last minimize.
    return (
        a[0] >= b[0]
        and a[1] >= b[1]
        and a[2] <= b[2]
        and (a[0] > b[0] or a[1] > b[1] or a[2] < b[2])
    )


def pareto_front_indices(objectives: List[Tuple[float, float, float]]) -> List[int]:
    n = len(objectives)
    front = []
    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[j], objectives[i]):
                is_dominated = True
                break
        if not is_dominated:
            front.append(i)
    return front


def scalar_fitness(obj: Tuple[float, float, float]) -> float:
    f1, fairness, comm = obj
    return 0.80 * f1 + 0.15 * fairness + 0.05 * (1.0 - comm)


def select_knee_point(
    candidate_indices: List[int],
    objectives: List[Tuple[float, float, float]],
) -> int:
    if len(candidate_indices) == 1:
        return candidate_indices[0]

    vals = np.array([objectives[i] for i in candidate_indices], dtype=np.float64)
    # Convert comm cost into utility to maximize.
    vals[:, 2] = 1.0 - vals[:, 2]

    mins = vals.min(axis=0)
    maxs = vals.max(axis=0)
    denom = np.where((maxs - mins) < 1e-12, 1.0, maxs - mins)
    norm = (vals - mins) / denom

    ideal = np.ones(3, dtype=np.float64)
    dists = np.linalg.norm(norm - ideal, axis=1)
    best_rel = int(np.argmin(dists))
    return candidate_indices[best_rel]


def decode_solution(
    vec: np.ndarray,
    n_clients: int,
    min_selected: int,
    max_selected: int,
) -> Tuple[List[int], np.ndarray]:
    sel_logits = vec[:n_clients]
    w_logits = vec[n_clients:]

    sel_probs = 1.0 / (1.0 + np.exp(-sel_logits))
    order = np.argsort(-sel_probs)
    k = max(min_selected, min(max_selected, n_clients))
    selected = list(order[:k])

    if len(selected) == 0:
        selected = [int(np.argmax(sel_probs))]

    raw_w = w_logits[selected]
    weights = softmax_np(raw_w)
    return selected, weights


def evaluate_candidate_solution(
    vec: np.ndarray,
    proxy_states: Sequence[Dict[str, torch.Tensor]],
    clients: Sequence[ClientData],
    input_dim: int,
    min_selected: int,
    max_selected: int,
    x_val_global: np.ndarray,
    y_val_global: np.ndarray,
    device: torch.device,
) -> Tuple[Tuple[float, float, float], List[int], np.ndarray]:
    n_clients = len(clients)
    selected, weights = decode_solution(vec, n_clients, min_selected, max_selected)

    client_quality = []
    for idx in selected:
        proxy_metrics = evaluate_state(
            model_state=proxy_states[idx],
            input_dim=input_dim,
            x=clients[idx].x_val,
            y=clients[idx].y_val,
            device=device,
        )
        client_quality.append(proxy_metrics["f1"])

    quality_prior = np.asarray(client_quality, dtype=np.float64)
    weights = softmax_np(np.log(weights + 1e-12) + 1.5 * quality_prior)

    sel_states = [proxy_states[i] for i in selected]
    agg_state = weighted_fedavg(sel_states, weights)

    global_metrics = evaluate_state(
        model_state=agg_state,
        input_dim=input_dim,
        x=x_val_global,
        y=y_val_global,
        device=device,
    )

    per_client_f1 = []
    for c in clients:
        m = evaluate_state(agg_state, input_dim=input_dim, x=c.x_val, y=c.y_val, device=device)
        per_client_f1.append(m["f1"])

    fairness = jains_fairness(per_client_f1)
    comm_cost = len(selected) / float(n_clients)
    obj = (global_metrics["f1"], fairness, comm_cost)
    return obj, selected, weights


def chaotic_sine_init(pop_size: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pop = np.zeros((pop_size, dim), dtype=np.float64)

    x = rng.uniform(low=0.01, high=0.99, size=dim)
    for i in range(pop_size):
        x = np.sin(np.pi * x)
        pop[i] = 8.0 * (x - 0.5)

    return pop


def build_greedy_seed_vector(
    clients: Sequence[ClientData],
    input_dim: int,
    k: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> np.ndarray:
    global_model = build_model(input_dim=input_dim, device=device)
    global_state = {name: tensor.detach().cpu().clone() for name, tensor in global_model.state_dict().items()}

    scores = []
    for i, c in enumerate(clients):
        state = train_local_model(
            global_state=global_state,
            input_dim=input_dim,
            x_train=c.x_train,
            y_train=c.y_train,
            epochs=1,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        metrics = evaluate_state(state, input_dim=input_dim, x=c.x_val, y=c.y_val, device=device)
        scores.append(float(metrics["f1"]))

    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    selected = list(order[: max(1, min(k, len(clients)))])

    vec = np.zeros(len(clients) * 2, dtype=np.float64)
    vec[: len(clients)] = -4.0
    vec[selected] = 4.0
    vec[len(clients) + np.array(selected)] = np.asarray([scores[i] for i in selected], dtype=np.float64)
    return vec


def run_mho_selection(
    proxy_states: Sequence[Dict[str, torch.Tensor]],
    clients: Sequence[ClientData],
    input_dim: int,
    x_val_global: np.ndarray,
    y_val_global: np.ndarray,
    mho_pop_size: int,
    mho_iters: int,
    min_selected: int,
    max_selected: int,
    device: torch.device,
    seed: int,
) -> Tuple[List[int], np.ndarray, Dict[str, float]]:
    n_clients = len(clients)
    dim = n_clients * 2

    pop = chaotic_sine_init(pop_size=mho_pop_size, dim=dim, seed=seed)
    greedy_seed = build_greedy_seed_vector(
        clients=clients,
        input_dim=input_dim,
        k=max(1, min(max_selected, n_clients)),
        batch_size=64,
        lr=1e-3,
        device=device,
    )
    pop[0] = greedy_seed
    rng = np.random.default_rng(seed + 123)

    objectives: List[Tuple[float, float, float]] = []
    decoded: List[Tuple[List[int], np.ndarray]] = []

    for i in range(mho_pop_size):
        obj, sel, w = evaluate_candidate_solution(
            vec=pop[i],
            proxy_states=proxy_states,
            clients=clients,
            input_dim=input_dim,
            min_selected=min_selected,
            max_selected=max_selected,
            x_val_global=x_val_global,
            y_val_global=y_val_global,
            device=device,
        )
        objectives.append(obj)
        decoded.append((sel, w))

    best_idx = int(np.argmax([scalar_fitness(o) for o in objectives]))

    for t in range(mho_iters):
        growth = 2.0 * (1.0 - (t / max(1, mho_iters - 1)))

        for i in range(mho_pop_size):
            r1, r2 = rng.choice(mho_pop_size, size=2, replace=False)
            x_i = pop[i]
            x_best = pop[best_idx]

            candidate = (
                x_i
                + growth * rng.random(dim) * (x_best - x_i)
                + growth * (rng.random(dim) - 0.5) * (pop[r1] - pop[r2])
            )
            reverse = -x_i

            cand_obj, cand_sel, cand_w = evaluate_candidate_solution(
                vec=candidate,
                proxy_states=proxy_states,
                clients=clients,
                input_dim=input_dim,
                min_selected=min_selected,
                max_selected=max_selected,
                x_val_global=x_val_global,
                y_val_global=y_val_global,
                device=device,
            )

            rev_obj, rev_sel, rev_w = evaluate_candidate_solution(
                vec=reverse,
                proxy_states=proxy_states,
                clients=clients,
                input_dim=input_dim,
                min_selected=min_selected,
                max_selected=max_selected,
                x_val_global=x_val_global,
                y_val_global=y_val_global,
                device=device,
            )

            curr_fit = scalar_fitness(objectives[i])
            cand_fit = scalar_fitness(cand_obj)
            rev_fit = scalar_fitness(rev_obj)

            if cand_fit >= curr_fit and cand_fit >= rev_fit:
                pop[i] = candidate
                objectives[i] = cand_obj
                decoded[i] = (cand_sel, cand_w)
            elif rev_fit >= curr_fit and rev_fit >= cand_fit:
                pop[i] = reverse
                objectives[i] = rev_obj
                decoded[i] = (rev_sel, rev_w)

        best_idx = int(np.argmax([scalar_fitness(o) for o in objectives]))

    front = pareto_front_indices(objectives)
    chosen_idx = select_knee_point(front, objectives)

    selected, weights = decoded[chosen_idx]
    f1, fairness, comm = objectives[chosen_idx]
    info = {
        "proxy_f1": float(f1),
        "proxy_fairness": float(fairness),
        "proxy_comm_cost": float(comm),
        "pareto_size": float(len(front)),
    }
    return selected, weights, info


def random_selection(n_clients: int, k: int, rng: np.random.Generator) -> Tuple[List[int], np.ndarray]:
    k = max(1, min(k, n_clients))
    selected = list(rng.choice(n_clients, size=k, replace=False))
    weights = np.ones(k, dtype=np.float64) / k
    return selected, weights


def greedy_selection(
    global_state: Dict[str, torch.Tensor],
    clients: Sequence[ClientData],
    input_dim: int,
    k: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> Tuple[List[int], np.ndarray]:
    local_scores = []
    for i, c in enumerate(clients):
        state = train_local_model(
            global_state=global_state,
            input_dim=input_dim,
            x_train=c.x_train,
            y_train=c.y_train,
            epochs=1,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        m = evaluate_state(state, input_dim=input_dim, x=c.x_val, y=c.y_val, device=device)
        local_scores.append((i, m["f1"]))

    local_scores.sort(key=lambda x: x[1], reverse=True)
    k = max(1, min(k, len(clients)))
    selected = [i for i, _ in local_scores[:k]]
    raw = np.array([max(1e-6, s) for _, s in local_scores[:k]], dtype=np.float64)
    weights = raw / np.sum(raw)
    return selected, weights


def clustered_selection(
    global_state: Dict[str, torch.Tensor],
    clients: Sequence[ClientData],
    input_dim: int,
    k: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> Tuple[List[int], np.ndarray]:
    n_clients = len(clients)
    k = max(1, min(k, n_clients))

    descriptors = []
    for c in clients:
        attack_ratio = float(np.mean(c.y_train))
        size = float(len(c.y_train))
        descriptors.append([attack_ratio, math.log1p(size)])

    desc = np.asarray(descriptors, dtype=np.float64)
    n_clusters = min(2, n_clients)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    cluster_ids = km.fit_predict(desc)

    local_f1 = []
    for i, c in enumerate(clients):
        state = train_local_model(
            global_state=global_state,
            input_dim=input_dim,
            x_train=c.x_train,
            y_train=c.y_train,
            epochs=1,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        m = evaluate_state(state, input_dim=input_dim, x=c.x_val, y=c.y_val, device=device)
        local_f1.append(float(m["f1"]))

    selected = []
    for cid in range(n_clusters):
        members = [i for i in range(n_clients) if cluster_ids[i] == cid]
        if not members:
            continue
        best = max(members, key=lambda i: local_f1[i])
        selected.append(best)

    if len(selected) < k:
        remaining = [i for i in range(n_clients) if i not in selected]
        remaining.sort(key=lambda i: local_f1[i], reverse=True)
        selected.extend(remaining[: (k - len(selected))])

    selected = selected[:k]
    raw = np.array([max(1e-6, local_f1[i]) for i in selected], dtype=np.float64)
    weights = raw / np.sum(raw)
    return selected, weights


def hybrid_selection(
    global_state: Dict[str, torch.Tensor],
    clients: Sequence[ClientData],
    input_dim: int,
    k: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> Tuple[List[int], np.ndarray]:
    n_clients = len(clients)
    k = max(1, min(k, n_clients))

    descriptors = []
    local_scores = []
    for i, c in enumerate(clients):
        attack_ratio = float(np.mean(c.y_train))
        size = float(len(c.y_train))
        descriptors.append([attack_ratio, math.log1p(size)])

        state = train_local_model(
            global_state=global_state,
            input_dim=input_dim,
            x_train=c.x_train,
            y_train=c.y_train,
            epochs=1,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        m = evaluate_state(state, input_dim=input_dim, x=c.x_val, y=c.y_val, device=device)
        local_scores.append(float(m["f1"]))

    desc = np.asarray(descriptors, dtype=np.float64)
    selected: List[int] = [int(np.argmax(local_scores))]

    if k > 1:
        centroid = desc.mean(axis=0)
        remaining = [i for i in range(n_clients) if i not in selected]
        remaining.sort(key=lambda i: np.linalg.norm(desc[i] - centroid), reverse=True)
        selected.extend(remaining[: (k - 1)])

    selected = selected[:k]
    weights = np.ones(len(selected), dtype=np.float64) / len(selected)
    return selected, weights


def run_federated_training(
    strategy: str,
    clients: Sequence[ClientData],
    input_dim: int,
    x_global_test: np.ndarray,
    y_global_test: np.ndarray,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    select_k: int,
    mho_pop_size: int,
    mho_iters: int,
    proxy_epochs: int,
    proxy_fraction: float,
    seed: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    global_model = build_model(input_dim=input_dim, device=device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    x_val_global = np.concatenate([c.x_val for c in clients], axis=0)
    y_val_global = np.concatenate([c.y_val for c in clients], axis=0)

    rounds_log = []
    rng = np.random.default_rng(seed + 333)

    for rnd in range(1, rounds + 1):
        if strategy == "all":
            selected = list(range(len(clients)))
            sizes = np.array([clients[i].n_train for i in selected], dtype=np.float64)
            weights = sizes / np.sum(sizes)
            extra = {}

        elif strategy == "random":
            selected, weights = random_selection(len(clients), select_k, rng=rng)
            extra = {}

        elif strategy == "greedy":
            selected, weights = greedy_selection(
                global_state=global_state,
                clients=clients,
                input_dim=input_dim,
                k=select_k,
                batch_size=batch_size,
                lr=lr,
                device=device,
                seed=seed + rnd,
            )
            extra = {}

        elif strategy == "clustered":
            selected, weights = clustered_selection(
                global_state=global_state,
                clients=clients,
                input_dim=input_dim,
                k=select_k,
                batch_size=batch_size,
                lr=lr,
                device=device,
                seed=seed + rnd,
            )
            extra = {}

        elif strategy == "mho":
            proxy_states = make_proxy_updates(
                global_state=global_state,
                clients=clients,
                input_dim=input_dim,
                proxy_epochs=proxy_epochs,
                proxy_fraction=proxy_fraction,
                batch_size=batch_size,
                lr=lr,
                device=device,
                seed=seed + rnd,
            )
            selected, weights, extra = run_mho_selection(
                proxy_states=proxy_states,
                clients=clients,
                input_dim=input_dim,
                x_val_global=x_val_global,
                y_val_global=y_val_global,
                mho_pop_size=mho_pop_size,
                mho_iters=mho_iters,
                min_selected=1,
                max_selected=max(1, min(select_k, len(clients))),
                device=device,
                seed=seed + rnd,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        local_states = []
        selected_train_sizes = []
        for idx in selected:
            c = clients[idx]
            state = train_local_model(
                global_state=global_state,
                input_dim=input_dim,
                x_train=c.x_train,
                y_train=c.y_train,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
            )
            local_states.append(state)
            selected_train_sizes.append(float(c.n_train))

        candidate_state = weighted_fedavg(local_states, weights)

        if strategy == "mho":
            # Hybrid-MHO aggregation: fuse optimizer weights with data-size priors,
            # then choose the most stable candidate on validation data.
            size_w = np.asarray(selected_train_sizes, dtype=np.float64)
            size_w = size_w / (np.sum(size_w) + 1e-12)
            mix_w = 0.8 * np.asarray(weights, dtype=np.float64) + 0.2 * size_w
            mix_w = mix_w / (np.sum(mix_w) + 1e-12)

            mixed_state = weighted_fedavg(local_states, mix_w)
            current_val = evaluate_state(global_state, input_dim=input_dim, x=x_val_global, y=y_val_global, device=device)
            candidate_val = evaluate_state(candidate_state, input_dim=input_dim, x=x_val_global, y=y_val_global, device=device)
            mixed_val = evaluate_state(mixed_state, input_dim=input_dim, x=x_val_global, y=y_val_global, device=device)

            stabilizer_alpha = 0.30 if candidate_val["f1"] < current_val["f1"] else 0.65
            stabilized_state = blend_model_states(global_state, candidate_state, alpha=stabilizer_alpha)
            stabilized_val = evaluate_state(stabilized_state, input_dim=input_dim, x=x_val_global, y=y_val_global, device=device)

            options = [
                (candidate_state, candidate_val, "mho_raw"),
                (mixed_state, mixed_val, "mho_mixed"),
                (stabilized_state, stabilized_val, "mho_stabilized"),
                (global_state, current_val, "mho_keep"),
            ]
            best_state, best_val, best_tag = max(options, key=lambda item: (item[1]["f1"], item[1]["auc"]))
            global_state = best_state
            extra["hmho_val_f1"] = float(best_val["f1"])
            extra["hmho_variant"] = best_tag
        else:
            global_state = candidate_state

        gmetrics = evaluate_state(global_state, input_dim=input_dim, x=x_global_test, y=y_global_test, device=device)
        per_client_f1 = []
        for c in clients:
            cm = evaluate_state(global_state, input_dim=input_dim, x=c.x_test, y=c.y_test, device=device)
            per_client_f1.append(cm["f1"])

        fairness = jains_fairness(per_client_f1)
        comm_cost = len(selected) / float(len(clients))

        row = {
            "round": rnd,
            "strategy": strategy,
            "accuracy": gmetrics["accuracy"],
            "f1": gmetrics["f1"],
            "auc": gmetrics["auc"],
            "fairness": fairness,
            "comm_cost": comm_cost,
            "selected_clients": len(selected),
            "selected_client_names": ";".join([clients[i].name for i in selected]),
        }
        for k, v in extra.items():
            row[k] = v
        rounds_log.append(row)

        print(
            f"[{strategy}] Round {rnd:03d} | F1={gmetrics['f1']:.4f} "
            f"Acc={gmetrics['accuracy']:.4f} AUC={gmetrics['auc']:.4f} "
            f"Fair={fairness:.4f} Comm={comm_cost:.2f}"
        )

    round_df = pd.DataFrame(rounds_log)
    final = round_df.iloc[-1].to_dict()
    summary = {
        "strategy": strategy,
        "final_accuracy": float(final["accuracy"]),
        "final_f1": float(final["f1"]),
        "final_auc": float(final["auc"]),
        "final_fairness": float(final["fairness"]),
        "avg_comm_cost": float(round_df["comm_cost"].mean()),
        "avg_selected_clients": float(round_df["selected_clients"].mean()),
    }
    return round_df, summary


def print_client_stats(clients: Sequence[ClientData]) -> None:
    print("\nClient distributions:")
    for c in clients:
        total = len(c.y_train) + len(c.y_val) + len(c.y_test)
        attack_ratio = float(np.mean(np.concatenate([c.y_train, c.y_val, c.y_test])))
        print(f"  {c.name}: total={total}, attack_ratio={attack_ratio:.4f}")


def parse_client_files(arg_clients: str) -> List[str]:
    if arg_clients.strip().lower() == "default":
        return list(DEFAULT_CLIENT_FILES)
    if arg_clients.strip().lower() == "all":
        return list(ALL_CLIENT_FILES)
    parts = [p.strip() for p in arg_clients.split(",") if p.strip()]
    if not parts:
        raise ValueError("No client files resolved from --clients argument.")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="FL-IDS with MHO-based client selection")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--clients", type=str, default="default", help="default | all | comma-separated list")
    parser.add_argument("--max-rows-per-client", type=int, default=50000)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--select-k", type=int, default=3)
    parser.add_argument("--mho-pop-size", type=int, default=60)
    parser.add_argument("--mho-iters", type=int, default=30)
    parser.add_argument("--proxy-epochs", type=int, default=1)
    parser.add_argument("--proxy-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategies", type=str, default="mho,random,greedy,clustered,all")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)

    client_files = parse_client_files(args.clients)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Selected client files: {client_files}")

    clients, input_dim, x_global_test, y_global_test = load_and_preprocess_clients(
        data_dir=args.data_dir,
        client_files=client_files,
        max_rows_per_client=args.max_rows_per_client,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    print(f"Input feature dimension after encoding: {input_dim}")
    print_client_stats(clients)

    strategies = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    all_round_logs = []
    summaries = []

    meta = {
        "data_dir": args.data_dir,
        "client_files": client_files,
        "config": vars(args),
        "device": str(device),
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for strategy in strategies:
        print(f"\n===== Running strategy: {strategy} =====")
        round_df, summary = run_federated_training(
            strategy=strategy,
            clients=clients,
            input_dim=input_dim,
            x_global_test=x_global_test,
            y_global_test=y_global_test,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            select_k=args.select_k,
            mho_pop_size=args.mho_pop_size,
            mho_iters=args.mho_iters,
            proxy_epochs=args.proxy_epochs,
            proxy_fraction=args.proxy_fraction,
            seed=args.seed,
            device=device,
        )
        all_round_logs.append(round_df)
        summaries.append(summary)

        round_path = os.path.join(run_dir, f"round_metrics_{strategy}.csv")
        round_df.to_csv(round_path, index=False)

    all_round_df = pd.concat(all_round_logs, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    all_round_df.to_csv(os.path.join(run_dir, "round_metrics_all.csv"), index=False)
    summary_df.to_csv(os.path.join(run_dir, "summary_comparison.csv"), index=False)

    print("\n===== Final Summary =====")
    print(summary_df.sort_values(by="final_f1", ascending=False).to_string(index=False))
    print(f"\nSaved outputs to: {run_dir}")


if __name__ == "__main__":
    main()
