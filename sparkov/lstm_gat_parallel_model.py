"""
lstm_gat_parallel_model.py — Parallel LSTM‖GAT hybrid for the Sparkov
credit-card-transaction dataset.

Sibling of lstm_gat_sequential_model.py — same reuse contract, same
duplicated sequence-construction helpers (kept file-local rather than
cross-imported, matching the fact that none of gatv2_model.py /
gcn_model.py / graphsage_model.py import from one another).

Architectural difference from the Sequential file: LSTM and GAT both read
the same cat/num-embedded input independently; merged via cross-attention
(GAT features as query, LSTM features as key/value) before classification,
instead of LSTM feeding into GAT as a pipeline stage.

Graph strategies: multi_relation | hybrid | intra_group
Run:  python lstm_gat_parallel_model.py
Outputs saved to ../outcomes/
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import average_precision_score

from config import CardFraudConfig
from utils import (
    set_seed, get_device, cleanup,
    load_and_preprocess,
    FoldPreprocessor, split_groups_holdout, build_stratified_folds,
    build_graph_edges, prepare_tensors, _make_sort_key,
    eval_from_probs, print_metrics, choose_threshold,
    save_results_plots, save_strategy_comparison,
    save_results_csv, print_final_comparison,
)

MODEL_ARCH = "lstm_gat_par"


# ============================================================================
#  SEQUENCE CONSTRUCTION  (duplicated from the Sequential file, see its
#  docstring for the rationale)
# ============================================================================
def _build_seq_indices(df_raw, cfg):
    """Chunks oversized groups instead of truncating -- see
    lstm_gat_sequential_model.py for why truncation was a real bug."""
    sort_key = _make_sort_key(df_raw, cfg.SORT_KEY_COLS)
    groups = []
    for _, idx in df_raw.groupby(cfg.INTRA_GROUP_KEY).groups.items():
        idx = list(idx)
        if len(idx) == 0:
            continue
        idx_sorted = sorted(idx, key=lambda i: sort_key[i])
        for start in range(0, len(idx_sorted), cfg.INTRA_MAX_GROUP_SIZE):
            groups.append(idx_sorted[start:start + cfg.INTRA_MAX_GROUP_SIZE])
    return groups


def _run_lstm_over_groups(lstm, x, groups, device):
    N = x.size(0)
    out_dim = lstm.hidden_size * (2 if lstm.bidirectional else 1)
    output = torch.zeros(N, out_dim, device=device)
    if not groups:
        return output
    lengths = [len(g) for g in groups]
    max_len = max(lengths)
    padded = torch.zeros(len(groups), max_len, x.size(1), device=device)
    for i, g in enumerate(groups):
        padded[i, :len(g)] = x[torch.as_tensor(g, device=device)]
    packed = nn.utils.rnn.pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
    lstm_out, _ = lstm(packed)
    unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
    for i, g in enumerate(groups):
        idx_t = torch.as_tensor(g, device=device)
        output[idx_t] = unpacked[i, :len(g)]
    return output


class CrossAttentionFusion(nn.Module):
    """GAT features query LSTM features; weighted LSTM value added residually to GAT."""

    def __init__(self, lstm_dim, gat_dim):
        super().__init__()
        self.query = nn.Linear(gat_dim, gat_dim)
        self.key   = nn.Linear(lstm_dim, gat_dim)
        self.value = nn.Linear(lstm_dim, gat_dim)
        # Zero-init: training starts mathematically identical to plain
        # GATv2 (fused ≈ gat_features at step 0); see IBM parallel model
        # for full rationale.
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        self.scale = gat_dim ** 0.5

    def forward(self, lstm_features, gat_features):
        Q = self.query(gat_features)
        K = self.key(lstm_features)
        V = self.value(lstm_features)
        attn = torch.softmax(torch.sum(Q * K, dim=1, keepdim=True) / self.scale, dim=0)
        return gat_features + attn * V


# ============================================================================
#  MODEL
# ============================================================================
class LSTMGATParFraudModel(nn.Module):
    def __init__(self, cardinalities, cat_cols, num_input_dim,
                 embedding_dim=8, hidden=64, heads=4,
                 lstm_hidden=None, lstm_layers=1, dropout=0.30):
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.dropout  = dropout
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(cardinalities[col] + 1, embedding_dim)
            for col in cat_cols})
        in_ch = len(cat_cols) * embedding_dim + num_input_dim

        lstm_hidden = lstm_hidden or 64
        self.lstm = nn.LSTM(
            input_size=in_ch, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden  # unidirectional: 1x, not 2x
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)

        self.gat1  = GATv2Conv(in_ch,          hidden, heads=heads, dropout=dropout)
        self.gat2  = GATv2Conv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden * heads)

        self.fusion = CrossAttentionFusion(lstm_out_dim, hidden * heads)

        self.cls = nn.Sequential(
            nn.Linear(hidden * heads, 128), nn.LeakyReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x_cat_dict, x_num, edge_index, groups):
        cat_embs = [self.embeddings[col](x_cat_dict[col]) for col in self.cat_cols]
        x = F.dropout(torch.cat([torch.cat(cat_embs, dim=1), x_num], dim=1),
                      p=self.dropout, training=self.training)

        lstm_feat = self.lstm_norm(_run_lstm_over_groups(self.lstm, x, groups, x.device))

        h = F.leaky_relu(self.norm1(self.gat1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        gat_feat = F.leaky_relu(self.norm2(self.gat2(h, edge_index)))

        fused = self.fusion(lstm_feat, gat_feat)
        fused = F.dropout(fused, p=self.dropout, training=self.training)
        return self.cls(fused).view(-1)


# ============================================================================
#  TRAIN LOOP
# ============================================================================
def train_loop_lstm(model, cat_tr_d, num_tr_t, y_tr_t, edge_tr, groups_tr,
                    cat_va_d, num_va_t, y_va_np, edge_va, groups_va,
                    cfg, verbose=False):
    pos = max(1, int(y_tr_t.sum().item()))
    neg = max(1, int(len(y_tr_t) - pos))
    pos_weight = torch.tensor([min(neg / pos, 20.0)], dtype=torch.float, device=y_tr_t.device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt  = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sch  = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-5)
    best_ap, best_state, no_improve = -1.0, None, 0

    @torch.no_grad()
    def val_ap():
        model.eval()
        probs = torch.sigmoid(model(cat_va_d, num_va_t, edge_va, groups_va)).cpu().numpy()
        return average_precision_score(y_va_np, probs) if len(np.unique(y_va_np)) == 2 else 0.0

    for ep in range(1, cfg.MAX_EPOCHS + 1):
        model.train(); opt.zero_grad()
        loss = crit(model(cat_tr_d, num_tr_t, edge_tr, groups_tr), y_tr_t)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(ep + 1e-8)
        if ep % cfg.EVAL_EVERY == 0:
            ap = val_ap()
            if ap > best_ap + 1e-6:
                best_ap, best_state, no_improve = ap, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
            if no_improve >= cfg.PATIENCE_CHECKS:
                if verbose: print(f"  Early stop at epoch {ep}")
                break
    if best_state is not None: model.load_state_dict(best_state)
    return model


# ============================================================================
#  FOLD TRAINING
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
    edge_tr = build_graph_edges(graph_strategy, df_tr, cat_tr, num_tr, cfg, verbose).to(device)
    edge_va = build_graph_edges(graph_strategy, df_va, cat_va, num_va, cfg, verbose).to(device)
    groups_tr = _build_seq_indices(df_tr.reset_index(drop=True), cfg)
    groups_va = _build_seq_indices(df_va.reset_index(drop=True), cfg)

    model = LSTMGATParFraudModel(
        cardinalities=prep.cardinalities, cat_cols=cfg.CATEGORICAL_COLS,
        num_input_dim=num_tr.shape[1], embedding_dim=cfg.EMBEDDING_DIM,
        hidden=cfg.HIDDEN_DIM, heads=cfg.HEADS, dropout=cfg.DROPOUT).to(device)

    model = train_loop_lstm(model, cat_tr_d, num_tr_t, y_tr_t, edge_tr, groups_tr,
                            cat_va_d, num_va_t, y_va_np, edge_va, groups_va, cfg, verbose)
    model.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(model(cat_tr_d, num_tr_t, edge_tr, groups_tr)).cpu().numpy()
        p_va = torch.sigmoid(model(cat_va_d, num_va_t, edge_va, groups_va)).cpu().numpy()
    thr = choose_threshold(y_va_np, p_va, cfg)
    cleanup()
    return {"model": model, "preprocessor": prep, "thr": thr,
            "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
            "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr)}


# ============================================================================
#  TEST ENSEMBLE
# ============================================================================
def run_test_ensemble_lstm(all_out, df_test, cfg, graph_strategy, device):
    test_probs = []
    df_test = df_test.reset_index(drop=True)
    for fold_i, out in enumerate(all_out):
        print(f"  Test fold {fold_i + 1}/{len(all_out)}...")
        prep = out["preprocessor"]
        df_te_enc = prep.transform(df_test)
        cat_te = df_te_enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
        num_te = df_te_enc[prep.num_cols].to_numpy(dtype=np.float32)
        edge_te = build_graph_edges(graph_strategy, df_test, cat_te, num_te, cfg).to(device)
        groups_te = _build_seq_indices(df_test, cfg)
        cat_te_d = {col: torch.tensor(cat_te[:, i], dtype=torch.long).to(device)
                    for i, col in enumerate(cfg.CATEGORICAL_COLS)}
        num_te_t = torch.tensor(num_te, dtype=torch.float).to(device)
        out["model"].eval()
        with torch.no_grad():
            probs = torch.sigmoid(out["model"](cat_te_d, num_te_t, edge_te, groups_te)).cpu().numpy()
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
def run_pipeline(df, cfg, graph_strategy="multi_relation"):
    set_seed(cfg.SEED); device = get_device()
    print(f"\n{'#'*60}\n# LSTM‖GAT (parallel) | {graph_strategy}\n{'#'*60}")
    df_dev, df_test = split_groups_holdout(
        df, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    folds = build_stratified_folds(df_dev, cfg.TARGET_COL, cfg.N_SPLITS,
                                   cfg.GROUP_KEY, cfg.STRATIFY_BINS, cfg.SEED)
    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"  Fold {i}/{cfg.N_SPLITS}")
        out = train_one_fold(df_tr, df_va, cfg, graph_strategy, i - 1, device)
        print_metrics("TRAIN", out["m_tr"]); print_metrics("VAL", out["m_va"]) 
        all_out.append(out); cleanup()
    p_ens, y_test, thr = run_test_ensemble_lstm(all_out, df_test, cfg, graph_strategy, device)
    m_test = eval_from_probs(y_test, p_ens, thr)
    print_metrics("TEST", m_test)
    result = {"all_out": all_out, "df_dev": df_dev, "df_test": df_test,
              "thr_global": thr, "test_metrics": m_test,
              "p_test_ens": p_ens, "y_test": y_test,
              "graph_strategy": graph_strategy, "model_arch": MODEL_ARCH}
    save_results_plots(result, cfg)
    return result


def run_all_strategies(df, cfg=None):
    if cfg is None: cfg = CardFraudConfig()
    results = {}
    for strat in ["multi_relation", "hybrid", "intra_group"]:
        results[f"{strat}_{MODEL_ARCH}"] = run_pipeline(df, cfg, strat)
        cleanup()
    print_final_comparison(results)
    save_strategy_comparison(results, cfg)
    save_results_csv(results, cfg)
    return results


if __name__ == "__main__":
    cfg = CardFraudConfig()
    df  = load_and_preprocess(train_path="fraudTrain.csv", test_path="fraudTest.csv", cfg=cfg)
    results = run_all_strategies(df, cfg)
