"""
Homogeneous account-level GAT fraud detector for the IBM dataset.

Unlike lstm_gat_sequential_model.py and lstm_gat_parallel_model.py, this file
does NOT add a new architecture -- it reuses GATFraudModel from
gatv2_model.py completely unchanged, together with utils.train_loop /
utils.safe_inference completely unchanged. What changes is the NODE
GRANULARITY: instead of one node per transaction, each node here is one
account (User), built by aggregating that account's full transaction
history into a single feature vector. This is still a homogeneous graph
(one node type, one merged edge set) -- it is simply a different topology
built on a different unit of analysis.

Because GATFraudModel's forward signature is (x_dense, x_high, x_low,
edge_index) with no sequence argument, this file needs no LSTM branch and
no custom train loop -- utils.train_loop / utils.safe_inference work as-is.
The only new code is:
  1. build_account_dataframe()   -- aggregates transactions -> one row/account
  2. AccountFraudConfig          -- a IBMFraudConfig subclass pointing
                                     DENSE/HIGH/LOW_CARD cols at the new
                                     aggregate columns
  3. build_account_graph_edges() -- account-to-account edges (this file's
                                     equivalent of build_graph_edges), with
                                     three strategies mirroring the paper's
                                     naming convention: similarity |
                                     shared_merchant | combined

Graph strategies: similarity | shared_merchant | combined
Run:  python account_gat_homogeneous_model.py
Outputs saved to ../outcomes/
"""

import gc
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from config import IBMFraudConfig
from utils import (
    set_seed, get_device, cleanup,
    load_and_preprocess,
    FoldPreprocessor, split_groups_holdout, build_group_stratified_folds,
    train_loop, safe_inference,
    eval_from_probs, print_metrics, choose_threshold,
    save_results_plots, save_strategy_comparison,
    save_results_csv, print_final_comparison,
)
from gatv2_model import GATFraudModel  # unchanged architecture, reused as-is

MODEL_ARCH = "account_gat"


# ============================================================================
#  CONFIG  (subclass -- only the column lists and graph knobs differ)
# ============================================================================
class AccountFraudConfig(IBMFraudConfig):
    GROUP_KEY: str = "User"          # one row per account already, but keep
                                      # the same split machinery (generic on
                                      # any key column) for consistency
    TARGET_COL: str = "is_fraud_account"

    HIGH_CARD_COLS = ["dominant_mcc", "dominant_merchant_state"]
    LOW_CARD_COLS  = ["dominant_use_chip"]
    DENSE_FEATURE_COLS = [
        "txn_count", "avg_amount", "std_amount", "max_amount", "total_amount",
        "unique_merchants", "unique_mcc", "merchant_diversity",
        "night_ratio", "weekend_ratio", "work_hour_ratio", "error_ratio",
    ]

    HIDDEN_DIM = 32   # kept identical to IBMFraudConfig for capacity parity
    HEADS = 2

    ACCOUNT_K_SIMILAR: int = 8
    ACCOUNT_SIM_THRESHOLD: float = 0.3
    ACCOUNT_MIN_SHARED_MERCHANTS: int = 2


# ============================================================================
#  ACCOUNT-LEVEL AGGREGATION  (the one genuinely new preprocessing step)
# ============================================================================
def build_account_dataframe(df_txn, cfg):
    """Collapse one row per transaction -> one row per account (User)."""
    print("Aggregating transactions into account-level features...")
    rows = []
    for user, g in df_txn.groupby("User"):
        night = g["hour"].between(0, 5, inclusive="both") | (g["hour"] >= 22)
        rows.append({
            "User": user,
            "txn_count": len(g),
            "avg_amount": g["Amount"].mean(),
            "std_amount": g["Amount"].std() if len(g) > 1 else 0.0,
            "max_amount": g["Amount"].max(),
            "total_amount": g["Amount"].sum(),
            "unique_merchants": g["Merchant Name"].nunique(),
            "unique_mcc": g["MCC"].nunique(),
            "merchant_diversity": g["Merchant Name"].nunique() / max(1, len(g)),
            "night_ratio": night.mean(),
            "weekend_ratio": g["is_weekend"].mean() if "is_weekend" in g else 0.0,
            "work_hour_ratio": g["is_work_hour"].mean() if "is_work_hour" in g else 0.0,
            "error_ratio": (g["Errors?"] != "None").mean() if "Errors?" in g else 0.0,
            "dominant_mcc": g["MCC"].mode().iloc[0] if len(g["MCC"].mode()) else -1,
            "dominant_merchant_state": (g["Merchant State"].mode().iloc[0]
                                         if "Merchant State" in g and len(g["Merchant State"].mode()) else "UNK"),
            "dominant_use_chip": (g["Use Chip"].mode().iloc[0]
                                   if "Use Chip" in g and len(g["Use Chip"].mode()) else "Unknown"),
            "is_fraud_account": int(g["Is Fraud?"].sum() > 0),
            "_merchant_set": frozenset(g["Merchant Name"].unique()),  # used only for graph building
        })
    df_acc = pd.DataFrame(rows).reset_index(drop=True)
    print(f"  {len(df_acc):,} accounts | fraud rate {df_acc['is_fraud_account'].mean():.4f}")
    return df_acc


# ============================================================================
#  ACCOUNT GRAPH CONSTRUCTION
# ============================================================================
def _similarity_edges(X_dense, X_high, X_low, cfg):
    feats = np.hstack([X_dense.astype("float32"), X_high.astype("float32"), X_low.astype("float32")])
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

def build_account_graph_edges(strategy, df_raw, X_dense, X_high, X_low, cfg, verbose=False):
    n = len(df_raw)
    if strategy == "similarity":
        edge_set = _similarity_edges(X_dense, X_high, X_low, cfg)
    elif strategy == "shared_merchant":
        edge_set = _shared_merchant_edges(df_raw, cfg)
    elif strategy == "combined":
        edge_set = _similarity_edges(X_dense, X_high, X_low, cfg)
        edge_set.update(_shared_merchant_edges(df_raw, cfg))
    else:
        raise ValueError(f"Unknown account graph strategy: '{strategy}'")
    for i in range(n): edge_set.add((i, i))  # self-loops, matching build_graph_edges
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    if verbose:
        deg = np.bincount(edge_index[0].cpu().numpy(), minlength=n)
        print(f"  [{strategy}] accounts={n:,} edges={edge_index.size(1):,} "
              f"deg(min/med/mean/max)=({deg.min()},{np.median(deg):.1f},{deg.mean():.1f},{deg.max()})")
    del edge_set; gc.collect()
    return edge_index


# ============================================================================
#  FOLD TRAINING  (reuses GATFraudModel + utils.train_loop/safe_inference
#  unchanged; only the edge builder and the absence of user-spending
#  features differ from gatv2_model.py's train_one_fold)
# ============================================================================
def train_one_fold(df_tr, df_va, cfg, graph_strategy, fold_idx=0, device=None):
    if device is None: device = get_device()
    prep = FoldPreprocessor(cfg); prep.fit(df_tr)
    df_tr_raw, df_va_raw = df_tr.reset_index(drop=True), df_va.reset_index(drop=True)
    X_d_tr, X_h_tr, X_l_tr, y_tr_np = prep.transform(df_tr_raw)
    X_d_va, X_h_va, X_l_va, y_va_np = prep.transform(df_va_raw)

    verbose = (fold_idx == 0)
    edge_tr = build_account_graph_edges(graph_strategy, df_tr_raw, X_d_tr, X_h_tr, X_l_tr, cfg, verbose)
    edge_va = build_account_graph_edges(graph_strategy, df_va_raw, X_d_va, X_h_va, X_l_va, cfg, verbose)
    del df_tr_raw, df_va_raw; gc.collect()

    x_d_tr = torch.tensor(X_d_tr, dtype=torch.float32).to(device)
    x_h_tr = torch.tensor(X_h_tr, dtype=torch.long).to(device)
    x_l_tr = torch.tensor(X_l_tr, dtype=torch.long).to(device)
    y_tr_t  = torch.tensor(y_tr_np, dtype=torch.float32).to(device)
    edge_tr = edge_tr.to(device)
    x_d_va = torch.tensor(X_d_va, dtype=torch.float32).to(device)
    x_h_va = torch.tensor(X_h_va, dtype=torch.long).to(device)
    x_l_va = torch.tensor(X_l_va, dtype=torch.long).to(device)
    edge_va = edge_va.to(device)
    del X_d_tr, X_h_tr, X_l_tr, X_d_va, X_h_va, X_l_va; gc.collect()

    model = GATFraudModel(
        dense_in_dim=x_d_tr.shape[1],
        high_card_cols=prep.high_card_cols, low_card_cols=prep.low_card_cols,
        factor_cardinalities=prep.factor_cardinalities,
        hidden=cfg.HIDDEN_DIM, heads=cfg.HEADS,
        dropout=cfg.DROPOUT,
        high_card_emb_dim=cfg.HIGH_CARD_EMB_DIM, low_card_emb_dim=cfg.LOW_CARD_EMB_DIM,
    ).to(device)

    use_amp = cfg.USE_AMP and device.type == "cuda"
    model = train_loop(model, x_d_tr, x_h_tr, x_l_tr, y_tr_t, edge_tr,
                       x_d_va, x_h_va, x_l_va, y_va_np, edge_va, cfg, device, verbose)
    model.eval()
    p_tr = safe_inference(model, x_d_tr, x_h_tr, x_l_tr, edge_tr, device, use_amp)
    p_va = safe_inference(model, x_d_va, x_h_va, x_l_va, edge_va, device, use_amp)
    thr  = choose_threshold(y_va_np, p_va, cfg)
    del x_d_tr, x_h_tr, x_l_tr, y_tr_t, edge_tr, x_d_va, x_h_va, x_l_va, edge_va; cleanup()
    return {"model": model.cpu(), "preprocessor": prep, "thr": thr,
             "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
             "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr)}


# ============================================================================
#  TEST ENSEMBLE  (simpler than the transaction-level version -- no
#  per-fold user-spending re-derivation needed, the account df already
#  contains that account's full aggregate history)
# ============================================================================
def run_test_ensemble_account(all_out, df_test, cfg, graph_strategy, device):
    test_probs = []
    use_amp = cfg.USE_AMP and device.type == "cuda"
    df_test = df_test.reset_index(drop=True)
    for fold_i, out in enumerate(all_out):
        print(f"  Test fold {fold_i+1}/{len(all_out)}...")
        prep = out["preprocessor"]
        X_d, X_h, X_l, y_te = prep.transform(df_test)
        edge_te = build_account_graph_edges(graph_strategy, df_test, X_d, X_h, X_l, cfg).to(device)
        x_d = torch.tensor(X_d, dtype=torch.float32).to(device)
        x_h = torch.tensor(X_h, dtype=torch.long).to(device)
        x_l = torch.tensor(X_l, dtype=torch.long).to(device)
        model = out["model"].to(device)
        probs = safe_inference(model, x_d, x_h, x_l, edge_te, device, use_amp)
        test_probs.append(probs); out["model"] = model.cpu()
        del x_d, x_h, x_l, edge_te; cleanup()
    p_ens = np.mean(test_probs, axis=0)
    thr_global = np.mean([o["thr"] for o in all_out])
    return p_ens, y_te, thr_global


# ============================================================================
#  PIPELINE
# ============================================================================
def run_pipeline(df_accounts, cfg, graph_strategy="combined"):
    set_seed(cfg.SEED); device = get_device()
    print(f"\n{'#'*60}\n# Homogeneous Account GAT | {graph_strategy}\n{'#'*60}")
    df_dev, df_test = split_groups_holdout(
        df_accounts, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    folds = build_group_stratified_folds(
        df_dev, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.N_SPLITS, cfg.STRATIFY_BINS, cfg.SEED)
    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"  Fold {i}/{cfg.N_SPLITS}")
        out = train_one_fold(df_tr, df_va, cfg, graph_strategy, i-1, device)
        print_metrics("TRAIN", out["m_tr"]); print_metrics("VAL", out["m_va"])
        all_out.append(out); cleanup()
    p_ens, y_test, thr = run_test_ensemble_account(all_out, df_test, cfg, graph_strategy, device)
    m_test = eval_from_probs(y_test, p_ens, thr); print_metrics("TEST", m_test)
    result = {"all_out": all_out, "df_dev": df_dev, "df_test": df_test,
               "thr_global": thr, "test_metrics": m_test,
               "p_test_ens": p_ens, "y_test": y_test,
               "graph_strategy": graph_strategy, "model_arch": MODEL_ARCH}
    save_results_plots(result, cfg)
    return result


def run_all_strategies(df_accounts, cfg=None):
    """Loops the account graph's own 3 strategies (similarity /
    shared_merchant / combined) -- the account-level analogue of
    multi_relation / hybrid / intra_group at the transaction level."""
    if cfg is None: cfg = AccountFraudConfig()
    results = {}
    for strat in ["similarity", "shared_merchant", "combined"]:
        results[f"{strat}_{MODEL_ARCH}"] = run_pipeline(df_accounts, cfg, strat); cleanup()
    print_final_comparison(results)
    save_strategy_comparison(results, cfg)
    save_results_csv(results, cfg)
    return results


if __name__ == "__main__":
    # Load with the base (transaction-level) config -- "Is Fraud?" still
    # exists at this point, "is_fraud_account" does not yet.
    load_cfg = IBMFraudConfig()
    df_txn = load_and_preprocess(path="/path/to/ibm_transactions.parquet", cfg=load_cfg)

    # Aggregate, THEN switch to the account-level config for the pipeline.
    cfg = AccountFraudConfig()
    df_accounts = build_account_dataframe(df_txn, cfg)
    del df_txn; gc.collect()
    results = run_all_strategies(df_accounts, cfg)
