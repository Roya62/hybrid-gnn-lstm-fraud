"""
account_gat_homogeneous_model.py — homogeneous account-level GAT for the
Sparkov dataset.

Like its IBM counterpart, this file adds NO new architecture — it reuses
GATv2FraudModel and utils.train_loop from gatv2_model.py / utils.py
completely unchanged. Only the NODE GRANULARITY changes: instead of one
node per transaction, each node here is one card (cc_num), built by
aggregating that card's full transaction history into a single feature
vector. Still a homogeneous graph (one node type, one merged edge set) —
just a different unit of analysis.

utils.run_test_ensemble is NOT reused as-is: it hardcodes the
transaction-level build_graph_edges() with strategy names
{multi_relation, hybrid, intra_group}, which don't apply here. This file's
run_test_ensemble_account() is a thin local copy that calls
build_account_graph_edges() instead — everything else about it (ensemble
averaging, threshold pooling) is identical to utils.run_test_ensemble.

Graph strategies: similarity | shared_merchant | combined
Run:  python account_gat_homogeneous_model.py
Outputs saved to ../outcomes/
"""

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from config import CardFraudConfig
from utils import (
    set_seed, get_device, cleanup,
    load_and_preprocess,
    FoldPreprocessor, split_groups_holdout, build_stratified_folds,
    prepare_tensors, train_loop,
    eval_from_probs, print_metrics, choose_threshold,
    save_results_plots, save_strategy_comparison,
    save_results_csv, print_final_comparison,
)
from gatv2_model import GATv2FraudModel  # unchanged architecture, reused as-is

MODEL_ARCH = "account_gat"


# ============================================================================
#  CONFIG
# ============================================================================
class CardAccountConfig(CardFraudConfig):
    GROUP_KEY: str = "cc_num"       # one row per account already
    TARGET_COL: str = "is_fraud_account"

    CATEGORICAL_COLS = ["dominant_category", "dominant_state", "gender"]
    NUMERICAL_COLS = [
        "txn_count", "avg_amt", "std_amt", "max_amt", "total_amt",
        "unique_merchants", "unique_categories", "merchant_diversity",
        "night_ratio", "weekend_ratio", "business_hours_ratio",
        "avg_distance", "age", "city_pop",
    ]

    HIDDEN_DIM = 64   # kept identical to CardFraudConfig for capacity parity
    HEADS = 4

    ACCOUNT_K_SIMILAR: int = 8
    ACCOUNT_SIM_THRESHOLD: float = 0.3
    ACCOUNT_MIN_SHARED_MERCHANTS: int = 2


# ============================================================================
#  ACCOUNT-LEVEL AGGREGATION
# ============================================================================
def build_account_dataframe(df_txn, cfg):
    """Collapse one row per transaction -> one row per card (cc_num)."""
    print("Aggregating transactions into account-level features...")
    has_distance = "distance" in df_txn.columns
    rows = []
    for cc, g in df_txn.groupby("cc_num"):
        night = (g["hour"] < 6) | (g["hour"] >= 22)
        rows.append({
            "cc_num": cc,
            "txn_count": len(g),
            "avg_amt": g["amt"].mean(),
            "std_amt": g["amt"].std() if len(g) > 1 else 0.0,
            "max_amt": g["amt"].max(),
            "total_amt": g["amt"].sum(),
            "unique_merchants": g["merchant"].nunique(),
            "unique_categories": g["category"].nunique(),
            "merchant_diversity": g["merchant"].nunique() / max(1, len(g)),
            "night_ratio": night.mean(),
            "weekend_ratio": (g["weekday"] >= 5).mean(),
            "business_hours_ratio": g["business_hours"].mean() if "business_hours" in g else 0.0,
            "avg_distance": g["distance"].mean() if has_distance else 0.0,
            "age": g["age"].iloc[0] if "age" in g else 0,
            "city_pop": g["city_pop"].iloc[0] if "city_pop" in g else 0,
            "dominant_category": g["category"].mode().iloc[0] if len(g["category"].mode()) else "unknown",
            "dominant_state": g["state"].mode().iloc[0] if "state" in g and len(g["state"].mode()) else "UNK",
            "gender": g["gender"].iloc[0] if "gender" in g else "U",
            "is_fraud_account": int(g["is_fraud"].sum() > 0),
            "_merchant_set": frozenset(g["merchant"].unique()),  # used only for graph building
        })
    df_acc = pd.DataFrame(rows).reset_index(drop=True)
    print(f"  {len(df_acc):,} accounts | fraud rate {df_acc['is_fraud_account'].mean():.4f}")
    return df_acc


# ============================================================================
#  ACCOUNT GRAPH CONSTRUCTION
# ============================================================================
def _similarity_edges(cat_np, num_np, cfg):
    feats = np.hstack([cat_np.astype("float32"), num_np.astype("float32")])
    norm = np.linalg.norm(feats, axis=1, keepdims=True); norm[norm == 0] = 1.0
    feats = feats / norm
    k = min(cfg.ACCOUNT_K_SIMILAR + 1, len(feats))
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(feats)
    dist, idx = nbrs.kneighbors(feats)
    edge_set = set()
    for i in range(len(feats)):
        for j, d in zip(idx[i, 1:], dist[i, 1:]):
            if (1 - d) >= cfg.ACCOUNT_SIM_THRESHOLD:
                edge_set.add((i, int(j))); edge_set.add((int(j), i))
    return edge_set

def _shared_merchant_edges(df_raw, cfg):
    edge_set = set()
    merchant_sets = df_raw["_merchant_set"].tolist()
    n = len(merchant_sets)
    for i in range(n):
        mi = merchant_sets[i]
        if not mi: continue
        for j in range(i + 1, n):
            if len(mi & merchant_sets[j]) >= cfg.ACCOUNT_MIN_SHARED_MERCHANTS:
                edge_set.add((i, j)); edge_set.add((j, i))
    return edge_set

def build_account_graph_edges(strategy, df_raw, cat_np, num_np, cfg, verbose=False):
    n = len(df_raw)
    if strategy == "similarity":
        edge_set = _similarity_edges(cat_np, num_np, cfg)
    elif strategy == "shared_merchant":
        edge_set = _shared_merchant_edges(df_raw, cfg)
    elif strategy == "combined":
        edge_set = _similarity_edges(cat_np, num_np, cfg)
        edge_set.update(_shared_merchant_edges(df_raw, cfg))
    else:
        raise ValueError(f"Unknown account graph strategy: '{strategy}'")
    for i in range(n): edge_set.add((i, i))  # self-loops
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    if verbose:
        deg = np.bincount(edge_index[0].cpu().numpy(), minlength=n)
        print(f"  [{strategy}] accounts={n:,} edges={edge_index.size(1):,} "
              f"deg(min/med/mean/max)=({deg.min()},{np.median(deg):.1f},{deg.mean():.1f},{deg.max()})")
    return edge_index


# ============================================================================
#  FOLD TRAINING  (reuses GATv2FraudModel + utils.train_loop unchanged;
#  only the edge builder differs from gatv2_model.py's train_one_fold)
# ============================================================================
def train_one_fold(df_tr, df_va, cfg, graph_strategy, fold_idx=0, device=None):
    if device is None: device = get_device()
    prep = FoldPreprocessor(cfg.CATEGORICAL_COLS, cfg.NUMERICAL_COLS)
    prep.fit(df_tr)
    df_tr_enc = prep.transform(df_tr)
    df_va_enc = prep.transform(df_va)

    cat_tr, num_tr, y_tr_np, cat_tr_d, num_tr_t = prepare_tensors(
        df_tr_enc, cfg.CATEGORICAL_COLS, prep.num_cols, cfg.TARGET_COL, device)
    cat_va, num_va, y_va_np, cat_va_d, num_va_t = prepare_tensors(
        df_va_enc, cfg.CATEGORICAL_COLS, prep.num_cols, cfg.TARGET_COL, device)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.float).to(device)

    verbose = (fold_idx == 0)
    df_tr_raw = df_tr.reset_index(drop=True)
    df_va_raw = df_va.reset_index(drop=True)
    edge_tr = build_account_graph_edges(graph_strategy, df_tr_raw, cat_tr, num_tr, cfg, verbose).to(device)
    edge_va = build_account_graph_edges(graph_strategy, df_va_raw, cat_va, num_va, cfg, verbose).to(device)

    model = GATv2FraudModel(
        cardinalities=prep.cardinalities, cat_cols=cfg.CATEGORICAL_COLS,
        num_input_dim=num_tr.shape[1], embedding_dim=cfg.EMBEDDING_DIM,
        hidden=cfg.HIDDEN_DIM, heads=cfg.HEADS, dropout=cfg.DROPOUT).to(device)

    model = train_loop(model, cat_tr_d, num_tr_t, y_tr_t, edge_tr,
                       cat_va_d, num_va_t, y_va_np, edge_va, cfg, verbose)
    model.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(model(cat_tr_d, num_tr_t, edge_tr)).cpu().numpy()
        p_va = torch.sigmoid(model(cat_va_d, num_va_t, edge_va)).cpu().numpy()
    thr = choose_threshold(y_va_np, p_va, cfg)
    cleanup()
    return {"model": model, "preprocessor": prep, "thr": thr,
            "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
            "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr)}


# ============================================================================
#  TEST ENSEMBLE  (local copy of utils.run_test_ensemble, using
#  build_account_graph_edges instead of the transaction-level builder)
# ============================================================================
def run_test_ensemble_account(all_out, df_test, cfg, graph_strategy, device):
    test_probs = []
    df_test = df_test.reset_index(drop=True)
    for fold_i, out in enumerate(all_out):
        print(f"  Test fold {fold_i + 1}/{len(all_out)}...")
        prep = out["preprocessor"]
        df_te_enc = prep.transform(df_test)
        cat_te = df_te_enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
        num_te = df_te_enc[prep.num_cols].to_numpy(dtype=np.float32)
        edge_te = build_account_graph_edges(graph_strategy, df_test, cat_te, num_te, cfg).to(device)
        cat_te_d = {col: torch.tensor(cat_te[:, i], dtype=torch.long).to(device)
                    for i, col in enumerate(cfg.CATEGORICAL_COLS)}
        num_te_t = torch.tensor(num_te, dtype=torch.float).to(device)
        out["model"].eval()
        with torch.no_grad():
            probs = torch.sigmoid(out["model"](cat_te_d, num_te_t, edge_te)).cpu().numpy()
        test_probs.append(probs); cleanup()
    p_ens = np.mean(np.vstack(test_probs), axis=0)
    y_test = df_test[cfg.TARGET_COL].astype(int).to_numpy()
    thr_global = choose_threshold(
        np.concatenate([o["y_va"] for o in all_out]),
        np.concatenate([o["p_va"] for o in all_out]), cfg)
    return p_ens, y_test, thr_global


# ============================================================================
#  PIPELINE
# ============================================================================
def run_pipeline(df_accounts, cfg, graph_strategy="combined"):
    set_seed(cfg.SEED); device = get_device()
    print(f"\n{'#'*60}\n# Homogeneous Account GAT | {graph_strategy}\n{'#'*60}")
    df_dev, df_test = split_groups_holdout(
        df_accounts, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    folds = build_stratified_folds(df_dev, cfg.TARGET_COL, cfg.N_SPLITS,
                                   cfg.GROUP_KEY, cfg.STRATIFY_BINS, cfg.SEED)
    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"  Fold {i}/{cfg.N_SPLITS}")
        out = train_one_fold(df_tr, df_va, cfg, graph_strategy, i - 1, device)
        print_metrics("TRAIN", out["m_tr"]); print_metrics("VAL", out["m_va"])
        all_out.append(out); cleanup()
    p_ens, y_test, thr = run_test_ensemble_account(all_out, df_test, cfg, graph_strategy, device)
    m_test = eval_from_probs(y_test, p_ens, thr)
    print_metrics("TEST", m_test)
    result = {"all_out": all_out, "df_dev": df_dev, "df_test": df_test,
              "thr_global": thr, "test_metrics": m_test,
              "p_test_ens": p_ens, "y_test": y_test,
              "graph_strategy": graph_strategy, "model_arch": MODEL_ARCH}
    save_results_plots(result, cfg)
    return result


def run_all_strategies(df_accounts, cfg=None):
    if cfg is None: cfg = CardAccountConfig()
    results = {}
    for strat in ["similarity", "shared_merchant", "combined"]:
        results[f"{strat}_{MODEL_ARCH}"] = run_pipeline(df_accounts, cfg, strat)
        cleanup()
    print_final_comparison(results)
    save_strategy_comparison(results, cfg)
    save_results_csv(results, cfg)
    return results


if __name__ == "__main__":
    # Load with the base (transaction-level) config -- "is_fraud" still
    # exists at this point, "is_fraud_account" does not yet.
    load_cfg = CardFraudConfig()
    df_txn = load_and_preprocess(train_path="fraudTrain.csv", test_path="fraudTest.csv", cfg=load_cfg)

    cfg = CardAccountConfig()
    df_accounts = build_account_dataframe(df_txn, cfg)
    del df_txn
    results = run_all_strategies(df_accounts, cfg)
