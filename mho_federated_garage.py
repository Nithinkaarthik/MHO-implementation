import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ClientData:
    client_id: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    attack_ratio: float


class MLPIDS(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MHOSelector:
    def __init__(
        self,
        num_clients: int,
        select_k: int,
        pop_size: int = 8,
        iterations: int = 4,
        mutation_prob: float = 0.08,
        seed: int = 42,
    ):
        self.num_clients = num_clients
        self.select_k = select_k
        self.pop_size = pop_size
        self.iterations = iterations
        self.mutation_prob = mutation_prob
        self.rng = np.random.default_rng(seed)
        self.population = self.rng.random((self.pop_size, self.num_clients * 2), dtype=np.float32)

    def _decode(self, vector: np.ndarray) -> Tuple[List[int], np.ndarray]:
        client_scores = vector[: self.num_clients]
        weight_scores = vector[self.num_clients :]

        selected = np.argsort(client_scores)[-self.select_k :].tolist()
        selected.sort()

        sel_weights = weight_scores[selected]
        exp_w = np.exp(sel_weights - np.max(sel_weights))
        exp_w = exp_w / (exp_w.sum() + 1e-12)
        return selected, exp_w

    def _update_population(self, ranked_population: np.ndarray, iteration: int) -> np.ndarray:
        best = ranked_population[0].copy()
        second = ranked_population[1].copy()
        next_pop = [best, second]

        exploration_scale = 0.55
        exploitation_scale = 0.12 * (1.0 - (iteration / max(self.iterations, 1)))

        while len(next_pop) < self.pop_size:
            idx = self.rng.integers(0, ranked_population.shape[0])
            current = ranked_population[idx].copy()

            if self.rng.random() < 0.5:
                peer = ranked_population[self.rng.integers(0, ranked_population.shape[0])]
                update = (
                    current
                    + exploration_scale * self.rng.random() * (best - current)
                    + exploration_scale * self.rng.random() * (peer - current)
                )
            else:
                noise = self.rng.normal(loc=0.0, scale=1.0, size=current.shape)
                update = best + exploitation_scale * noise

            mutation_mask = self.rng.random(update.shape[0]) < self.mutation_prob
            if mutation_mask.any():
                update[mutation_mask] = self.rng.random(np.count_nonzero(mutation_mask))

            update = np.clip(update, 0.0, 1.0)
            next_pop.append(update.astype(np.float32))

        return np.stack(next_pop, axis=0)

    def optimize(
        self,
        global_state: Dict[str, torch.Tensor],
        model_ctor,
        clients: List[ClientData],
        x_server_val: np.ndarray,
        y_server_val: np.ndarray,
        device: torch.device,
        local_lr: float,
        batch_size: int,
    ) -> Tuple[List[int], np.ndarray, Dict[str, float]]:
        fitnesses = []
        decoded = []

        for iteration_idx in range(self.iterations):
            fitnesses.clear()
            decoded.clear()

            for i in range(self.population.shape[0]):
                selected, weights = self._decode(self.population[i])
                new_state, _round_info = simulate_round(
                    global_state=global_state,
                    model_ctor=model_ctor,
                    clients=clients,
                    selected_clients=selected,
                    selection_weights=weights,
                    device=device,
                    local_epochs=1,
                    local_lr=local_lr,
                    batch_size=batch_size,
                    max_batches=8,
                )

                model = model_ctor().to(device)
                model.load_state_dict(new_state)
                val_metrics = evaluate_model(model, x_server_val, y_server_val, device)

                client_f1s = []
                for cid in selected:
                    cm = evaluate_model(model, clients[cid].x_val.cpu().numpy(), clients[cid].y_val.cpu().numpy(), device)
                    client_f1s.append(cm["f1"])

                fairness_var = float(np.var(client_f1s)) if client_f1s else 1.0
                comm_cost = len(selected) / float(self.num_clients)

                # Multi-objective scalarization: lower is better
                objective = 0.6 * (1.0 - val_metrics["f1"]) + 0.25 * fairness_var + 0.15 * comm_cost

                fitnesses.append(objective)
                decoded.append((selected, weights, val_metrics["f1"], fairness_var, comm_cost))

            ranked_idx = np.argsort(np.array(fitnesses))
            ranked_population = self.population[ranked_idx]
            self.population = self._update_population(ranked_population, iteration=iteration_idx)

        final_fitness = []
        final_decoded = []
        for i in range(self.population.shape[0]):
            selected, weights = self._decode(self.population[i])
            new_state, _ = simulate_round(
                global_state=global_state,
                model_ctor=model_ctor,
                clients=clients,
                selected_clients=selected,
                selection_weights=weights,
                device=device,
                local_epochs=1,
                local_lr=local_lr,
                batch_size=batch_size,
                max_batches=8,
            )
            model = model_ctor().to(device)
            model.load_state_dict(new_state)
            val_metrics = evaluate_model(model, x_server_val, y_server_val, device)

            client_f1s = []
            for cid in selected:
                cm = evaluate_model(model, clients[cid].x_val.cpu().numpy(), clients[cid].y_val.cpu().numpy(), device)
                client_f1s.append(cm["f1"])
            fairness_var = float(np.var(client_f1s)) if client_f1s else 1.0
            comm_cost = len(selected) / float(self.num_clients)
            objective = 0.6 * (1.0 - val_metrics["f1"]) + 0.25 * fairness_var + 0.15 * comm_cost

            final_fitness.append(objective)
            final_decoded.append((selected, weights, val_metrics["f1"], fairness_var, comm_cost))

        best_idx = int(np.argmin(np.array(final_fitness)))
        best_selected, best_weights, best_f1, best_fairness, best_comm = final_decoded[best_idx]
        debug_info = {
            "proxy_f1": float(best_f1),
            "fairness_var": float(best_fairness),
            "comm_cost": float(best_comm),
            "fitness": float(final_fitness[best_idx]),
        }
        return best_selected, best_weights, debug_info


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_garage_data(csv_path: str, sample_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, low_memory=False)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)

    df["date"] = df["date"].astype(str).str.strip()
    df["time"] = df["time"].astype(str).str.strip()
    datetime_text = df["date"] + " " + df["time"]
    datetime_col = pd.to_datetime(datetime_text, format="%d-%b-%y %H:%M:%S", errors="coerce")
    datetime_col = datetime_col.fillna(datetime_col.mode().iloc[0])

    df["hour"] = datetime_col.dt.hour
    df["minute"] = datetime_col.dt.minute
    df["weekday"] = datetime_col.dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)

    # Keep attack type only for non-IID partitioning, not as model input.
    attack_type = df["type"].astype(str).values
    y = df["label"].astype(int).values

    x_df = df[["door_state", "sphone_signal", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]].copy()
    x_df = pd.get_dummies(x_df, columns=["door_state", "sphone_signal"], drop_first=False)

    x_train, x_test, y_train, y_test, type_train, _ = train_test_split(
        x_df.values,
        y,
        attack_type,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, type_train, attack_type


def create_non_iid_clients(
    x_train: np.ndarray,
    y_train: np.ndarray,
    attack_types_train: np.ndarray,
    num_clients: int,
    alpha: float,
    val_size: float,
    seed: int,
) -> List[ClientData]:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(y_train))
    client_indices = {cid: [] for cid in range(num_clients)}

    unique_types = sorted(np.unique(attack_types_train).tolist())

    for t_idx, attack in enumerate(unique_types):
        attack_idx = all_indices[attack_types_train == attack]
        rng.shuffle(attack_idx)

        probs = rng.dirichlet(np.ones(num_clients) * alpha)
        dominant_client = t_idx % num_clients
        probs[dominant_client] += 1.25
        probs = probs / probs.sum()

        splits = (np.cumsum(probs) * len(attack_idx)).astype(int)
        chunks = np.split(attack_idx, splits[:-1])

        for cid, chunk in enumerate(chunks):
            if chunk.size > 0:
                client_indices[cid].extend(chunk.tolist())

    # Guarantee each client has enough samples.
    min_required = 64
    donor_clients = sorted(client_indices.keys(), key=lambda c: len(client_indices[c]), reverse=True)
    for cid in range(num_clients):
        if len(client_indices[cid]) >= min_required:
            continue
        needed = min_required - len(client_indices[cid])
        for donor in donor_clients:
            if donor == cid or len(client_indices[donor]) <= min_required + 8:
                continue
            take = min(needed, len(client_indices[donor]) - (min_required + 8))
            if take <= 0:
                continue
            transfer = client_indices[donor][-take:]
            client_indices[donor] = client_indices[donor][:-take]
            client_indices[cid].extend(transfer)
            needed -= take
            if needed <= 0:
                break

    clients: List[ClientData] = []
    for cid in range(num_clients):
        idx = np.array(client_indices[cid], dtype=np.int64)
        rng.shuffle(idx)

        x_local = x_train[idx]
        y_local = y_train[idx]

        if len(np.unique(y_local)) < 2:
            split_point = max(int(len(idx) * (1 - val_size)), 1)
            tr_idx = np.arange(split_point)
            va_idx = np.arange(split_point, len(idx)) if split_point < len(idx) else np.array([split_point - 1])
            x_tr, y_tr = x_local[tr_idx], y_local[tr_idx]
            x_va, y_va = x_local[va_idx], y_local[va_idx]
        else:
            x_tr, x_va, y_tr, y_va = train_test_split(
                x_local,
                y_local,
                test_size=val_size,
                random_state=seed + cid,
                stratify=y_local,
            )

        clients.append(
            ClientData(
                client_id=cid,
                x_train=torch.tensor(x_tr, dtype=torch.float32),
                y_train=torch.tensor(y_tr, dtype=torch.float32),
                x_val=torch.tensor(x_va, dtype=torch.float32),
                y_val=torch.tensor(y_va, dtype=torch.float32),
                attack_ratio=float(np.mean(y_local)),
            )
        )

    return clients


def train_local(
    global_state: Dict[str, torch.Tensor],
    model_ctor,
    client: ClientData,
    device: torch.device,
    local_epochs: int,
    local_lr: float,
    batch_size: int,
    max_batches: int = -1,
) -> Tuple[Dict[str, torch.Tensor], int]:
    model = model_ctor().to(device)
    model.load_state_dict(global_state)
    model.train()

    ds = TensorDataset(client.x_train, client.y_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    pos_count = float(torch.sum(client.y_train).item())
    neg_count = float(len(client.y_train) - pos_count)
    if pos_count > 0:
        pos_weight = torch.tensor([max(neg_count / pos_count, 1.0)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=local_lr)

    for _ in range(local_epochs):
        for b_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            if max_batches > 0 and (b_idx + 1) >= max_batches:
                break

    trained_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return trained_state, len(client.y_train)


def weighted_fedavg(local_states: List[Dict[str, torch.Tensor]], weights: np.ndarray) -> Dict[str, torch.Tensor]:
    agg_state = {}
    for k in local_states[0].keys():
        stacked = torch.stack([state[k] * float(weights[i]) for i, state in enumerate(local_states)], dim=0)
        agg_state[k] = torch.sum(stacked, dim=0)
    return agg_state


def simulate_round(
    global_state: Dict[str, torch.Tensor],
    model_ctor,
    clients: List[ClientData],
    selected_clients: List[int],
    selection_weights: np.ndarray,
    device: torch.device,
    local_epochs: int,
    local_lr: float,
    batch_size: int,
    max_batches: int = -1,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    local_states = []
    sample_sizes = []

    for idx, cid in enumerate(selected_clients):
        state, n_samples = train_local(
            global_state=global_state,
            model_ctor=model_ctor,
            client=clients[cid],
            device=device,
            local_epochs=local_epochs,
            local_lr=local_lr,
            batch_size=batch_size,
            max_batches=max_batches,
        )
        local_states.append(state)
        sample_sizes.append(n_samples)

    sample_sizes = np.array(sample_sizes, dtype=np.float64)
    selection_weights = np.array(selection_weights, dtype=np.float64)

    # Combine optimizer-provided weights with data-size weighting for stability.
    agg_w = selection_weights * sample_sizes
    agg_w = agg_w / (agg_w.sum() + 1e-12)

    new_state = weighted_fedavg(local_states, agg_w)
    info = {
        "selected_clients": len(selected_clients),
        "avg_client_samples": float(np.mean(sample_sizes)),
    }
    return new_state, info


def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(x_t).squeeze(1).detach().cpu().numpy()

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }

    if len(np.unique(y)) > 1:
        metrics["auc"] = float(roc_auc_score(y, probs))
    else:
        metrics["auc"] = 0.5
    return metrics


def evaluate_per_client(model: nn.Module, clients: List[ClientData], device: torch.device) -> Dict[str, float]:
    f1s = []
    for c in clients:
        m = evaluate_model(model, c.x_val.cpu().numpy(), c.y_val.cpu().numpy(), device)
        f1s.append(m["f1"])

    return {
        "client_f1_mean": float(np.mean(f1s)),
        "client_f1_std": float(np.std(f1s)),
        "client_f1_var": float(np.var(f1s)),
    }


def model_size_bytes(state: Dict[str, torch.Tensor]) -> int:
    total = 0
    for v in state.values():
        total += v.numel() * v.element_size()
    return int(total)


def rounds_to_threshold(history: List[float], threshold: float) -> int:
    for i, v in enumerate(history, start=1):
        if v >= threshold:
            return i
    return len(history)


def run_strategy(
    strategy_name: str,
    base_state: Dict[str, torch.Tensor],
    model_ctor,
    clients: List[ClientData],
    x_server_val: np.ndarray,
    y_server_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int,
    local_epochs: int,
    local_lr: float,
    batch_size: int,
    num_selected_clients: int,
    device: torch.device,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    global_state = copy.deepcopy(base_state)
    param_bytes = model_size_bytes(global_state)
    num_clients = len(clients)
    mho_selector = MHOSelector(
        num_clients=num_clients,
        select_k=num_selected_clients,
        pop_size=8,
        iterations=4,
        mutation_prob=0.08,
        seed=seed + 99,
    )

    history = []
    comm_total_bytes = 0.0
    last_client_scores = np.zeros(num_clients, dtype=np.float32)

    for rnd in range(rounds):
        model = model_ctor().to(device)
        model.load_state_dict(global_state)

        local_f1 = []
        for c in clients:
            cm = evaluate_model(model, c.x_val.cpu().numpy(), c.y_val.cpu().numpy(), device)
            local_f1.append(cm["f1"])
        last_client_scores = np.array(local_f1, dtype=np.float32)

        if strategy_name == "all_clients_fedavg":
            selected = list(range(num_clients))
            weights = np.ones(num_clients, dtype=np.float64) / num_clients
        elif strategy_name == "random":
            selected = sorted(rng.choice(num_clients, size=num_selected_clients, replace=False).tolist())
            weights = np.ones(len(selected), dtype=np.float64) / len(selected)
        elif strategy_name == "greedy":
            sample_sizes = np.array([len(c.y_train) for c in clients], dtype=np.float32)
            size_norm = sample_sizes / (sample_sizes.max() + 1e-12)
            greedy_score = 0.7 * (1.0 - last_client_scores) + 0.3 * size_norm
            selected = np.argsort(greedy_score)[-num_selected_clients:].tolist()
            selected.sort()
            weights = np.ones(len(selected), dtype=np.float64) / len(selected)
        elif strategy_name == "cluster":
            sample_sizes = np.array([len(c.y_train) for c in clients], dtype=np.float32)
            sample_norm = sample_sizes / (sample_sizes.max() + 1e-12)
            meta = np.column_stack(
                [
                    sample_norm,
                    np.array([c.attack_ratio for c in clients], dtype=np.float32),
                    last_client_scores,
                ]
            )

            n_clusters = min(num_selected_clients, num_clients)
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed + rnd, n_init=10)
            labels = kmeans.fit_predict(meta)
            selected = []
            for cluster_id in range(n_clusters):
                members = np.where(labels == cluster_id)[0]
                if len(members) == 0:
                    continue
                # Pick the most underperforming client in each cluster.
                chosen = members[np.argmin(last_client_scores[members])]
                selected.append(int(chosen))

            while len(selected) < num_selected_clients:
                pool = [i for i in range(num_clients) if i not in selected]
                if not pool:
                    break
                selected.append(pool[0])

            selected = sorted(selected[:num_selected_clients])
            weights = np.ones(len(selected), dtype=np.float64) / len(selected)
        elif strategy_name == "mho":
            selected, weights, _ = mho_selector.optimize(
                global_state=global_state,
                model_ctor=model_ctor,
                clients=clients,
                x_server_val=x_server_val,
                y_server_val=y_server_val,
                device=device,
                local_lr=local_lr,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        new_state, _ = simulate_round(
            global_state=global_state,
            model_ctor=model_ctor,
            clients=clients,
            selected_clients=selected,
            selection_weights=np.array(weights, dtype=np.float64),
            device=device,
            local_epochs=local_epochs,
            local_lr=local_lr,
            batch_size=batch_size,
        )
        global_state = new_state

        model = model_ctor().to(device)
        model.load_state_dict(global_state)

        test_metrics = evaluate_model(model, x_test, y_test, device)
        fairness_metrics = evaluate_per_client(model, clients, device)

        comm_total_bytes += len(selected) * param_bytes

        history.append(
            {
                "round": rnd + 1,
                "selected_clients": selected,
                "test": test_metrics,
                "fairness": fairness_metrics,
                "comm_MB_cumulative": comm_total_bytes / (1024.0 * 1024.0),
            }
        )

        print(
            f"[{strategy_name}] round={rnd + 1:02d} f1={test_metrics['f1']:.4f} "
            f"auc={test_metrics['auc']:.4f} fairness_var={fairness_metrics['client_f1_var']:.6f} "
            f"comm_MB={comm_total_bytes / (1024.0 * 1024.0):.2f}"
        )

    final_f1 = history[-1]["test"]["f1"]
    f1_curve = [h["test"]["f1"] for h in history]
    threshold = 0.95 * max(f1_curve)

    return {
        "strategy": strategy_name,
        "history": history,
        "final": history[-1],
        "convergence_round_95pct_peak_f1": rounds_to_threshold(f1_curve, threshold),
    }


def save_outputs(results: Dict[str, Dict[str, object]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    summary = {}
    for name, payload in results.items():
        final = payload["final"]
        summary[name] = {
            "final_test": final["test"],
            "final_fairness": final["fairness"],
            "communication_MB": final["comm_MB_cumulative"],
            "convergence_round_95pct_peak_f1": payload["convergence_round_95pct_peak_f1"],
        }

    with open(os.path.join(output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        for name, payload in results.items():
            rounds = [h["round"] for h in payload["history"]]
            f1s = [h["test"]["f1"] for h in payload["history"]]
            plt.plot(rounds, f1s, marker="o", label=name)
        plt.xlabel("Federated Round")
        plt.ylabel("Global Test F1-score")
        plt.title("Convergence Comparison on ToN-IoT Garage Door")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "convergence_f1.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        for name, payload in results.items():
            rounds = [h["round"] for h in payload["history"]]
            comm = [h["comm_MB_cumulative"] for h in payload["history"]]
            plt.plot(rounds, comm, marker="s", label=name)
        plt.xlabel("Federated Round")
        plt.ylabel("Cumulative Communication (MB)")
        plt.title("Communication Overhead Comparison")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "communication_overhead.png"), dpi=150)
        plt.close()
    except Exception as exc:
        print(f"Warning: plotting skipped due to: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MHO-optimized Federated IDS on ToN-IoT Garage Door dataset")
    parser.add_argument("--data", type=str, default="IoT_Garage_Door.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-fraction", type=float, default=0.10)
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument("--num-selected-clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--local-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.35)
    parser.add_argument("--output-dir", type=str, default="outputs_garage")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train, x_test, y_train, y_test, attack_types_train, _ = preprocess_garage_data(
        csv_path=args.data,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )

    x_train_main, x_server_val, y_train_main, y_server_val, attack_main, _ = train_test_split(
        x_train,
        y_train,
        attack_types_train,
        test_size=0.15,
        random_state=args.seed,
        stratify=y_train,
    )

    clients = create_non_iid_clients(
        x_train=x_train_main,
        y_train=y_train_main,
        attack_types_train=attack_main,
        num_clients=args.num_clients,
        alpha=args.dirichlet_alpha,
        val_size=0.2,
        seed=args.seed,
    )

    input_dim = x_train.shape[1]

    def model_ctor():
        return MLPIDS(input_dim=input_dim)

    base_model = model_ctor().to(device)
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}

    strategies = ["all_clients_fedavg", "random", "greedy", "cluster", "mho"]
    results = {}

    for i, strategy in enumerate(strategies):
        print(f"\nRunning strategy: {strategy}")
        results[strategy] = run_strategy(
            strategy_name=strategy,
            base_state=base_state,
            model_ctor=model_ctor,
            clients=clients,
            x_server_val=x_server_val,
            y_server_val=y_server_val,
            x_test=x_test,
            y_test=y_test,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            local_lr=args.local_lr,
            batch_size=args.batch_size,
            num_selected_clients=min(args.num_selected_clients, args.num_clients),
            device=device,
            seed=args.seed + i * 13,
        )

    save_outputs(results, args.output_dir)

    print("\nFinal Summary:")
    for strategy, payload in results.items():
        final = payload["final"]
        print(
            f"{strategy:>18} | f1={final['test']['f1']:.4f} auc={final['test']['auc']:.4f} "
            f"fair_var={final['fairness']['client_f1_var']:.6f} commMB={final['comm_MB_cumulative']:.2f} "
            f"conv95={payload['convergence_round_95pct_peak_f1']}"
        )


if __name__ == "__main__":
    main()
