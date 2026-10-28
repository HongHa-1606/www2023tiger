import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


DAY = 24 * 60 * 60
TAU_GRID = [1 * DAY, 7 * DAY, 30 * DAY, 90 * DAY]
WINDOW_GRID = [1 * DAY, 7 * DAY, 30 * DAY, 90 * DAY, 180 * DAY]
RANDOM_SEEDS = [17, 29, 43, 71, 101]
HISTORICAL_SEED = 2026


def _find_checkpoint(project_root):
    results_dir = project_root / "results"
    saved_models_dir = project_root / "saved_models"
    candidates = []
    for result_path in results_dir.glob("*.json"):
        with result_path.open(encoding="utf-8") as file:
            config = json.load(file)
        checkpoint = saved_models_dir / f"{result_path.stem}.pth"
        if config.get("data") == "reddit" and checkpoint.is_file():
            candidates.append((result_path.stat().st_mtime, result_path, checkpoint, config))
    if not candidates:
        raise FileNotFoundError(
            f"No trained Reddit checkpoint found under {saved_models_dir}. "
            "Run the TIGER training cell first and keep results/ and saved_models/."
        )
    _, result_path, checkpoint, config = max(candidates, key=lambda item: item[0])
    return result_path, checkpoint, config


def _model_score_vector(model, collator, event, candidate_ids, device, item_offset):
    user_id = int(event.u)
    positive_id = int(event.i)
    dummy_negative = int(candidate_ids[0] if candidate_ids[0] != positive_id else candidate_ids[1])
    timestamp = float(event.ts)
    edge_id = int(event.event_id) + 1
    batch = [
        (user_id, int(candidate_id), dummy_negative, timestamp, edge_id, 0)
        for candidate_id in candidate_ids
    ]
    src_ids, dst_ids, neg_ids, ts, eids, _, comp_graph = collator(batch)
    comp_graph.to(device)
    src_ids = src_ids.to(device)
    dst_ids = dst_ids.to(device)
    neg_ids = neg_ids.to(device)
    ts = ts.to(device)
    eids = eids.to(device)

    outdated_ids, messages, message_ts = model.compute_messages(
        comp_graph.np_computation_graph_nodes
    )
    involved_ids = comp_graph.computation_graph_nodes
    involved_reprs = model.right_memory.vals[involved_ids].clone()
    if len(outdated_ids):
        updated_reprs = model.apply_messages(outdated_ids, messages, message_ts)
        involved_idx, outdated_idx = torch.where(involved_ids[:, None] == outdated_ids)
        involved_reprs[involved_idx] = updated_reprs[outdated_idx]

    batch_node_ids = torch.cat((src_ids, dst_ids, neg_ids))
    representations = model.compute_temporal_embedding_with_involved_nodes_only(
        involved_reprs, batch_node_ids, ts.repeat(3), comp_graph
    )
    batch_size = len(src_ids)
    src_repr, dst_repr, _ = representations.reshape(3, batch_size, model.nfeat_dim)
    src_hits, dst_hits, _, _ = comp_graph.hit_data
    if model.hit_type == "vec":
        src_pair = torch.cat((src_repr, src_hits), dim=1)
        dst_pair = torch.cat((dst_repr, dst_hits), dim=1)
    elif model.hit_type == "bin":
        src_pair = src_repr + model.hit_embedding(src_hits.max(1).values.long())
        dst_pair = dst_repr + model.hit_embedding(dst_hits.max(1).values.long())
    elif model.hit_type == "count":
        src_pair = src_repr + model.hit_embedding(src_hits.sum(1).long())
        dst_pair = dst_repr + model.hit_embedding(dst_hits.sum(1).long())
    else:
        src_pair, dst_pair = src_repr, dst_repr
    scores = model.score_fn(src_pair, dst_pair).squeeze(1)
    result = scores.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"Non-finite TIGER scores at event {event.event_id}.")
    return result


def _update_model(model, collator, event, device, dummy_negative):
    batch = [(
        int(event.u),
        int(event.i),
        int(dummy_negative),
        float(event.ts),
        int(event.event_id) + 1,
        0,
    )]
    src_ids, dst_ids, neg_ids, ts, eids, _, comp_graph = collator(batch)
    comp_graph.to(device)
    model.contrast_learning(
        src_ids.to(device),
        dst_ids.to(device),
        neg_ids.to(device),
        ts.to(device),
        eids.to(device),
        comp_graph,
    )


def _rank_metrics(scores):
    positive = scores[0]
    negatives = scores[1:]
    rank = 1.0 + np.sum(negatives > positive) + 0.5 * np.sum(negatives == positive)
    return float(rank), float(1.0 / rank)


def _time_week(timestamp):
    return pd.Timestamp(int(timestamp), unit="s", tz="UTC").strftime("%G-W%V")


def _new_baseline_state(item_count):
    return {
        "popularity": np.zeros(item_count, dtype=np.float64),
        "decayed": {tau: np.zeros(item_count, dtype=np.float64) for tau in TAU_GRID},
        "last_global_ts": None,
        "last_pair_ts": {},
        "prior_by_user": defaultdict(set),
        "seen_pairs": set(),
    }


def _prepare_baselines(state, user_id, positive_id, timestamp, all_subreddit_ids, item_offset):
    previous_ts = state["last_global_ts"]
    if previous_ts is not None:
        delta = max(0.0, timestamp - previous_ts)
        for tau in TAU_GRID:
            state["decayed"][tau] *= math.exp(-delta / tau)
    state["last_global_ts"] = timestamp
    popularity = state["popularity"]
    vectors = {
        "Popularity": popularity.copy(),
        "EdgeBank-unlimited": np.asarray([
            float((user_id, int(candidate)) in state["last_pair_ts"])
            for candidate in all_subreddit_ids
        ]),
    }
    vectors.update({
        f"Popularity-decay-{tau}": vector.copy()
        for tau, vector in state["decayed"].items()
    })
    vectors.update({
        f"EdgeBank-window-{window}": np.asarray([
            float(
                (user_id, int(candidate)) in state["last_pair_ts"]
                and timestamp - state["last_pair_ts"][(user_id, int(candidate))] <= window
            )
            for candidate in all_subreddit_ids
        ])
        for window in WINDOW_GRID
    })
    return vectors


def _update_baselines(state, user_id, positive_id, timestamp, item_index):
    state["last_pair_ts"][(user_id, positive_id)] = timestamp
    state["prior_by_user"][user_id].add(positive_id)
    state["seen_pairs"].add((user_id, positive_id))
    idx = item_index(positive_id)
    state["popularity"][idx] += 1.0
    for tau in TAU_GRID:
        state["decayed"][tau][idx] += 1.0


def _candidate_pools(positive_id, prior_items, all_subreddit_ids):
    random_candidates = [
        int(candidate) for candidate in all_subreddit_ids
        if int(candidate) != positive_id and int(candidate) not in prior_items
    ]
    historical_candidates = [
        int(candidate) for candidate in all_subreddit_ids
        if int(candidate) != positive_id and int(candidate) in prior_items
    ]
    return {
        "random": np.asarray([positive_id] + random_candidates, dtype=np.int64),
        "historical": np.asarray([positive_id] + historical_candidates, dtype=np.int64),
    }


def run_evaluation(project_root, output_dir):
    from init_utils import init_model
    from tiger.data.data_loader import GraphCollator, InteractionData
    from tiger.data.graph import Graph

    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint_path, config = _find_checkpoint(project_root)
    seed = int(config.get("seed", 0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = project_root / "data" / "ml_reddit.csv"
    graph_df = pd.read_csv(data_path)
    if not {"u", "i", "ts", "label", "idx"}.issubset(graph_df.columns):
        raise ValueError(f"{data_path} is missing the TIGER interaction columns.")
    nfeats = np.load(project_root / "data" / "ml_reddit_node.npy", allow_pickle=False)
    efeats = np.load(project_root / "data" / "ml_reddit.npy", allow_pickle=False)
    if efeats.ndim != 2 or efeats.shape[0] != len(graph_df) + 1:
        raise ValueError("TIGER edge features must include the padding row and every event.")
    full_data = InteractionData(
        graph_df.u.to_numpy(dtype=np.int64),
        graph_df.i.to_numpy(dtype=np.int64),
        graph_df.ts.to_numpy(dtype=np.float64),
        graph_df.idx.to_numpy(dtype=np.int64),
        graph_df.label.to_numpy(dtype=np.int64),
        eval=False,
    )
    train_events = pd.read_csv(output_dir / "train_events.csv", usecols=["event_id"])
    train_ids = set(train_events.event_id.to_numpy(dtype=np.int64) + 1)
    train_mask = graph_df.idx.isin(train_ids).to_numpy()
    if not train_mask.any():
        raise ValueError("Stage 4 train split contains no events.")
    train_data = InteractionData(
        graph_df.loc[train_mask, "u"].to_numpy(dtype=np.int64),
        graph_df.loc[train_mask, "i"].to_numpy(dtype=np.int64),
        graph_df.loc[train_mask, "ts"].to_numpy(dtype=np.float64),
        graph_df.loc[train_mask, "idx"].to_numpy(dtype=np.int64),
        graph_df.loc[train_mask, "label"].to_numpy(dtype=np.int64),
        eval=False,
    )
    strategy = config.get("strategy", "recent_edges")
    train_graph = Graph.from_data(train_data, strategy=strategy, seed=seed)
    full_graph = Graph.from_data(full_data, strategy=strategy, seed=seed)
    model = init_model(
        nfeats,
        efeats,
        train_graph,
        full_graph,
        full_data,
        device,
        feature_as_buffer=bool(config.get("feature_as_buffer", True)),
        dim=config.get("dim"),
        n_layers=int(config.get("n_layers", 1)),
        n_heads=int(config.get("n_heads", 2)),
        n_neighbors=int(config.get("n_neighbors", 10)),
        hit_type=config.get("hit_type", "bin"),
        dropout=float(config.get("dropout", 0.1)),
        restarter_type=config.get("restarter_type", "static"),
        hist_len=int(config.get("hist_len", 40)),
        msg_src=config.get("msg_src", "left"),
        upd_src=config.get("upd_src", "right"),
        msg_tsfm_type=config.get("msg_tsfm_type", "id"),
        mem_update_type=config.get("mem_update_type", "gru"),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.graph = full_graph
    model.eval()
    model.reset()
    collator = GraphCollator(
        full_graph,
        int(config.get("n_neighbors", 10)),
        int(config.get("n_layers", 1)),
        restarter=config.get("restarter_type", "static"),
        hist_len=int(config.get("hist_len", 40)),
    )

    events = pd.read_csv(project_root / "data" / "ml_reddit.csv")
    if not {"u", "i", "ts", "idx"}.issubset(events.columns):
        raise ValueError("data/ml_reddit.csv is missing the TIGER interaction columns.")
    event_splits = []
    for split_name in ("train", "val", "test"):
        split_path = output_dir / f"{split_name}_events.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing Stage 4 event split: {split_path}")
        split_df = pd.read_csv(split_path, usecols=["event_id"])
        split_df["split"] = split_name
        event_splits.append(split_df)
    split_map = pd.concat(event_splits, ignore_index=True)
    if split_map["event_id"].nunique() != len(events):
        raise ValueError("Stage 4 event files do not cover each TIGER event exactly once.")
    events["event_id"] = events["idx"].astype(np.int64) - 1
    events = events.merge(split_map, on="event_id", how="left", validate="one_to_one")
    if events["split"].isna().any():
        raise ValueError("Some TIGER interactions have no Stage 4 split.")
    events = events.sort_values(["ts", "idx"], kind="mergesort").reset_index(drop=True)
    node_map_path = output_dir / "stage4_node_map.csv"
    if not node_map_path.is_file():
        raise FileNotFoundError(
            f"Missing {node_map_path}; rerun the Stage 4 notebook cells to generate node mapping."
        )
    node_map = pd.read_csv(node_map_path)
    subreddit_offset = int(node_map.loc[node_map.node_type.eq("subreddit"), "node_id"].min())
    all_subreddit_ids = np.sort(
        node_map.loc[node_map.node_type.eq("subreddit"), "node_id"].to_numpy(dtype=np.int64)
    )
    if len(all_subreddit_ids) < 2:
        raise ValueError("At least two subreddit nodes are required for negative evaluation.")

    item_count = len(all_subreddit_ids)
    item_offset = subreddit_offset

    def item_index(node_id):
        idx = int(node_id) - item_offset
        if idx < 0 or idx >= item_count:
            raise ValueError(f"Subreddit node ID {node_id} is outside the mapped range.")
        return idx

    val_tau_rr = defaultdict(list)
    val_window_rr = defaultdict(list)
    tune_state = _new_baseline_state(item_count)
    for row in events.itertuples(index=False):
        split_name = str(row.split)
        if split_name == "test":
            break
        user_id = int(row.u)
        positive_id = int(row.i)
        timestamp = float(row.ts)
        prior_items = tune_state["prior_by_user"][user_id]
        is_new = (user_id, positive_id) not in tune_state["seen_pairs"]
        baseline_vectors = _prepare_baselines(
            tune_state, user_id, positive_id, timestamp, all_subreddit_ids, item_offset
        )
        if split_name == "val":
            modes = ["all"] + (["new"] if is_new else [])
            candidate_pools = _candidate_pools(positive_id, prior_items, all_subreddit_ids)
            for mode in modes:
                for kind, candidates in candidate_pools.items():
                    if len(candidates) < 2:
                        continue
                    candidate_idx = candidates - item_offset
                    for tau in TAU_GRID:
                        rr = _rank_metrics(
                            baseline_vectors[f"Popularity-decay-{tau}"][candidate_idx]
                        )[1]
                        val_tau_rr[(mode, kind, tau)].append(rr)
                    for window in WINDOW_GRID:
                        rr = _rank_metrics(
                            baseline_vectors[f"EdgeBank-window-{window}"][candidate_idx]
                        )[1]
                        val_window_rr[(mode, kind, window)].append(rr)
        _update_baselines(tune_state, user_id, positive_id, timestamp, item_index)

    def mean_validation_score(results, parameter):
        groups = [
            float(np.mean(values))
            for (mode, kind, value), values in results.items()
            if value == parameter and values
        ]
        if not groups:
            raise RuntimeError("No validation events had an available candidate negative.")
        return float(np.mean(groups))

    best_tau = max(TAU_GRID, key=lambda value: mean_validation_score(val_tau_rr, value))
    best_window = max(WINDOW_GRID, key=lambda value: mean_validation_score(val_window_rr, value))
    parameter_rows = []
    for tau in TAU_GRID:
        parameter_rows.append({
            "model": "Popularity-decay",
            "parameter": "tau_seconds",
            "value": tau,
            "validation_mrr": mean_validation_score(val_tau_rr, tau),
            "selected": tau == best_tau,
        })
    for window in WINDOW_GRID:
        parameter_rows.append({
            "model": "EdgeBank-window",
            "parameter": "window_seconds",
            "value": window,
            "validation_mrr": mean_validation_score(val_window_rr, window),
            "selected": window == best_window,
        })
    pd.DataFrame(parameter_rows).to_csv(
        output_dir / "stage6_validation_parameters.csv", index=False
    )
    print(f"Validation selected tau={best_tau / DAY:g} days, W={best_window / DAY:g} days.")

    rank_rows = defaultdict(list)
    pair_rows = defaultdict(lambda: {"positive": [], "negative": []})
    random_rngs = {random_seed: np.random.default_rng(random_seed) for random_seed in RANDOM_SEEDS}
    historical_rng = np.random.default_rng(HISTORICAL_SEED)
    test_state = _new_baseline_state(item_count)

    model_names = [
        "TIGER",
        "Popularity",
        "Popularity-decay",
        "EdgeBank-unlimited",
        "EdgeBank-window",
    ]

    with torch.inference_mode():
        for row in events.itertuples(index=False):
            user_id = int(row.u)
            positive_id = int(row.i)
            timestamp = float(row.ts)
            split_name = str(row.split)
            pair = (user_id, positive_id)
            prior_items = test_state["prior_by_user"][user_id]
            is_new = pair not in test_state["seen_pairs"]
            positive_index = item_index(positive_id)
            baseline_vectors = _prepare_baselines(
                test_state, user_id, positive_id, timestamp, all_subreddit_ids, item_offset
            )
            candidate_pools = _candidate_pools(positive_id, prior_items, all_subreddit_ids)

            if split_name == "test":
                tiger_scores = _model_score_vector(
                    model, collator, row, all_subreddit_ids, device, item_offset
                )
                model_vectors = {
                    "TIGER": tiger_scores,
                    "Popularity": baseline_vectors["Popularity"],
                    "Popularity-decay": baseline_vectors[f"Popularity-decay-{best_tau}"],
                    "EdgeBank-unlimited": baseline_vectors["EdgeBank-unlimited"],
                    "EdgeBank-window": baseline_vectors[f"EdgeBank-window-{best_window}"],
                }
                modes = ["all"] + (["new"] if is_new else [])
                sampled_negatives = {}
                for kind, candidate_ids in candidate_pools.items():
                    if len(candidate_ids) < 2:
                        continue
                    if kind == "random":
                        sampled_negatives[kind] = {
                            random_seed: int(rng.choice(candidate_ids[1:]))
                            for random_seed, rng in random_rngs.items()
                        }
                    else:
                        sampled_negatives[kind] = {
                            HISTORICAL_SEED: int(historical_rng.choice(candidate_ids[1:]))
                        }
                for mode in modes:
                    for kind, candidate_ids in candidate_pools.items():
                        if len(candidate_ids) < 2:
                            continue
                        candidate_idx = candidate_ids - subreddit_offset
                        week = _time_week(timestamp)
                        for model_name in model_names:
                            scores = model_vectors[model_name][candidate_idx]
                            rank, reciprocal_rank = _rank_metrics(scores)
                            rank_rows[(model_name, mode, kind)].append({
                                "week": week,
                                "mrr": reciprocal_rank,
                                "hits1": float(rank <= 1),
                                "hits5": float(rank <= 5),
                                "hits10": float(rank <= 10),
                            })
                            positive_score = float(model_vectors[model_name][positive_index])
                            if kind == "random":
                                for random_seed, negative_id in sampled_negatives[kind].items():
                                    key = (model_name, mode, kind, random_seed)
                                    pair_rows[key]["positive"].append(positive_score)
                                    pair_rows[key]["negative"].append(
                                        float(model_vectors[model_name][item_index(negative_id)])
                                    )
                            else:
                                negative_id = sampled_negatives[kind][HISTORICAL_SEED]
                                key = (model_name, mode, kind, HISTORICAL_SEED)
                                pair_rows[key]["positive"].append(positive_score)
                                pair_rows[key]["negative"].append(
                                    float(model_vectors[model_name][item_index(negative_id)])
                                )

            dummy_negative = int(
                all_subreddit_ids[0] if all_subreddit_ids[0] != positive_id
                else all_subreddit_ids[1]
            )
            _update_model(model, collator, row, device, dummy_negative)
            _update_baselines(
                test_state, user_id, positive_id, timestamp, item_index
            )

    metric_rows = []
    for mode in ("new", "all"):
        for kind in ("random", "historical"):
            for model_name in model_names:
                ranking = rank_rows.get((model_name, mode, kind), [])
                seeds = RANDOM_SEEDS if kind == "random" else [HISTORICAL_SEED]
                aucs = []
                aps = []
                pair_count = 0
                for random_seed in seeds:
                    samples = pair_rows.get((model_name, mode, kind, random_seed))
                    if not samples or not samples["positive"]:
                        continue
                    scores = np.asarray(samples["positive"] + samples["negative"])
                    labels = np.concatenate((
                        np.ones(len(samples["positive"]), dtype=np.int8),
                        np.zeros(len(samples["negative"]), dtype=np.int8),
                    ))
                    aucs.append(float(roc_auc_score(labels, scores)))
                    aps.append(float(average_precision_score(labels, scores)))
                    pair_count = max(pair_count, len(samples["positive"]))
                metric_rows.append({
                    "Model": model_name,
                    "Evaluation": f"Test-{mode}",
                    "Negative type": kind,
                    "AUC": float(np.mean(aucs)) if aucs else np.nan,
                    "AUC_std": float(np.std(aucs)) if aucs else np.nan,
                    "AP": float(np.mean(aps)) if aps else np.nan,
                    "AP_std": float(np.std(aps)) if aps else np.nan,
                    "MRR": float(np.mean([row["mrr"] for row in ranking])) if ranking else np.nan,
                    "Hits@1": float(np.mean([row["hits1"] for row in ranking])) if ranking else np.nan,
                    "Hits@5": float(np.mean([row["hits5"] for row in ranking])) if ranking else np.nan,
                    "Hits@10": float(np.mean([row["hits10"] for row in ranking])) if ranking else np.nan,
                    "rank_events": len(ranking),
                    "pair_events": pair_count,
                    "negative_seeds": len(aucs),
                })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "stage6_test_metrics.csv", index=False)
    weekly_rows = []
    for (model_name, mode, kind), values in rank_rows.items():
        for row in values:
            weekly_rows.append({
                "Model": model_name,
                "Evaluation": f"Test-{mode}",
                "Negative type": kind,
                "week": row["week"],
                "MRR": row["mrr"],
            })
    weekly = pd.DataFrame(weekly_rows)
    weekly_summary = weekly.groupby(
        ["Model", "Evaluation", "Negative type", "week"], as_index=False
    )["MRR"].mean()
    weekly_summary.to_csv(output_dir / "stage6_weekly_mrr.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    for axis, (mode, kind) in zip(
        axes.flat,
        [("new", "random"), ("new", "historical"), ("all", "random"), ("all", "historical")],
    ):
        part = weekly_summary[
            weekly_summary["Evaluation"].eq(f"Test-{mode}")
            & weekly_summary["Negative type"].eq(kind)
        ]
        for model_name in model_names:
            curve = part[part["Model"].eq(model_name)].sort_values("week")
            if not curve.empty:
                axis.plot(curve["week"], curve["MRR"], marker="o", label=model_name)
        axis.set_title(f"Test-{mode} / {kind}")
        axis.set_xlabel("Test week (UTC)")
        axis.set_ylabel("MRR")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[0].legend(handles, labels, loc="best", fontsize="small")
    fig.tight_layout()
    chart_path = output_dir / "stage6_weekly_mrr.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    print(f"Checkpoint: {checkpoint_path} ({result_path.name})")
    print(metrics.to_string(index=False))
    print(f"Metrics: {output_dir / 'stage6_test_metrics.csv'}")
    print(f"Weekly MRR: {output_dir / 'stage6_weekly_mrr.csv'}")
    print(f"Chart: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="Online Stage 6 evaluation for TIGER and baselines.")
    parser.add_argument("--root", required=True, help="Writable TIGER repository root.")
    parser.add_argument("--output-dir", required=True, help="Stage 4/6 artifact directory.")
    args = parser.parse_args()
    run_evaluation(Path(args.root), Path(args.output_dir))


if __name__ == "__main__":
    main()
