"""
lstm_only_model.py — solo LSTM baseline for the IBM dataset (no graph
component at all).

Sibling of gatv2_model.py / lstm_gat_sequential_model.py / lstm_gat_parallel_model.py.
This is the missing ablation corner: the hybrid results (Sequential,
Parallel) show what happens when a temporal branch is ADDED to a fixed
graph topology; this file shows what a temporal branch achieves with NO
relational signal at all. Comparing this against the GATv2-only baseline
isolates the graph's contribution the same way the hybrid files isolate
the LSTM's contribution.

Because this model has no edge_index and never touches graph structure,
graph_strategy is not an experimental variable here — the model is
identical regardless of which topology φ would have been used, so it is
trained and evaluated exactly ONCE per dataset, not once per strategy.
Running it three times under three strategy labels would just be three
identical models differing only by random seed, which is not a
meaningful comparison axis.

Reuses unchanged from utils.py: FoldPreprocessor, split_groups_holdout,
build_group_stratified_folds, choose_threshold, eval_from_probs,
print_metrics, save_results_plots, save_strategy_comparison,
save_results_csv, print_final_comparison. Sequence construction
(_build_seq_indices, _run_lstm_over_groups) and the numerical-stability
choices (full-coverage chunking, LayerNorm, capped pos_weight) are
carried over unchanged from lstm_gat_sequential_model.py.

Run:  python lstm_only_model.py
Outputs saved to ../outcomes/
"""

import gc
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score

from config import IBMFraudConfig
from utils import (
    set_seed, get_device, cleanup,
    load_and_preprocess, add_user_spending_features,
    FoldPreprocessor, split_groups_holdout, build_group_stratified_folds,
    _make_sort_key,
    eval_from_probs, print_metrics, choose_threshold,
    save_results_plots, save_strategy_comparison,
    save_results_csv, print_final_comparison,
)

MODEL_ARCH = "lstm_only"


# ============================================================================
#  SEQUENCE CONSTRUCTION (identical to lstm_gat_sequential_model.py)
# ============================================================================
def _build_seq_indices(df_raw, cfg):
    """Chunks oversized per-cardholder groups into windows instead of
    truncating, so every node gets a genuine LSTM pass. See
    lstm_gat_sequential_model.py for the full rationale."""
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
    """Causal: every node gets the LSTM's output at its OWN position in
    its cardholder's chronological sequence, not a broadcast final state."""
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
        # Explicit dtype cast: under AMP, `unpacked` may be Half while
        # `output` is float32; advanced/fancy indexing requires an exact
        # dtype match in PyTorch (basic slicing above casts implicitly,
        # this line does not).
        output[idx_t] = unpacked[i, :len(g)].to(output.dtype)
    return output


# ============================================================================
#  MODEL — no graph component anywhere
# ============================================================================
class LSTMOnlyFraudModel(nn.Module):
    """encode() copied verbatim from GATFraudModel so this baseline sees
    identical input construction to the GATv2 baseline and both hybrids —
    the ONLY difference in this whole comparison family is what happens
    after encode(): here, nothing but LSTM -> classifier, no graph stage
    at all."""

    def __init__(self, dense_in_dim, high_card_cols, low_card_cols,
                 factor_cardinalities, hidden, heads=2,
                 lstm_hidden=None, lstm_layers=1,
                 dropout=0.30, high_card_emb_dim=8, low_card_emb_dim=4):
        super().__init__()
        self.dropout        = dropout
        self.high_card_cols = list(high_card_cols)
        self.low_card_cols  = list(low_card_cols)
        self.high_emb = nn.ModuleDict({
            col: nn.Embedding(int(factor_cardinalities[col]) + 2, high_card_emb_dim)
            for col in high_card_cols})
        self.low_emb = nn.ModuleDict({
            col: nn.Embedding(int(factor_cardinalities[col]) + 2, low_card_emb_dim)
            for col in low_card_cols})
        in_dim = (dense_in_dim + len(high_card_cols)*high_card_emb_dim
                  + len(low_card_cols)*low_card_emb_dim)

        # lstm_hidden chosen so bidirectional output == hidden*heads,
        # i.e. the SAME dimensionality the GATv2 baseline's final layer
        # produces before its classifier head (Section 3.4 of the paper).
        # This keeps the classifier head architecture identical across
        # baseline / LSTM-only / Sequential / Parallel, isolating the
        # encoder stage as the only varying component.
        lstm_hidden = lstm_hidden or (hidden * heads)
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden  # unidirectional: 1x, not 2x
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)

        # Same 2-layer MLP head shape as GATFraudModel / hybrid models.
        self.cls = nn.Sequential(
            nn.Linear(lstm_out_dim, 128), nn.LeakyReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def encode(self, x_dense, x_high, x_low):
        parts = [x_dense]
        if self.high_card_cols:
            parts.append(torch.cat([self.high_emb[col](x_high[:, i]+1)
                                    for i, col in enumerate(self.high_card_cols)], dim=1))
        if self.low_card_cols:
            parts.append(torch.cat([self.low_emb[col](x_low[:, i]+1)
                                    for i, col in enumerate(self.low_card_cols)], dim=1))
        return torch.cat(parts, dim=1)

    def forward(self, x_dense, x_high, x_low, groups):
        x = F.dropout(self.encode(x_dense, x_high, x_low), p=self.dropout, training=self.training)
        x = self.lstm_norm(_run_lstm_over_groups(self.lstm, x, groups, x.device))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.cls(x).view(-1)


# ============================================================================
#  SAFE INFERENCE / TRAIN LOOP  (no edge_index anywhere)
# ============================================================================
@torch.no_grad()
def safe_inference_lstm(model, x_d, x_h, x_l, groups, device, use_amp=True):
    model.eval()
    try:
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(x_d, x_h, x_l, groups)
        return torch.sigmoid(logits).cpu().numpy()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("  [WARN] GPU OOM — falling back to CPU")
            cleanup(); model_cpu = model.cpu()
            logits = model_cpu(x_d.cpu(), x_h.cpu(), x_l.cpu(), groups)
            model.to(device); return torch.sigmoid(logits).numpy()
        raise


def train_loop_lstm(model, x_d_tr, x_h_tr, x_l_tr, y_tr_t, groups_tr,
                    x_d_va, x_h_va, x_l_va, y_va_np, groups_va,
                    cfg, device, verbose=False):
    pos = max(1, int(y_tr_t.sum().item())); neg = max(1, int(len(y_tr_t)-pos))
    # Capped at 20x rather than raw neg/pos ratio -- see
    # lstm_gat_sequential_model.py for why an uncapped weight can collapse
    # a slower-converging recurrent model under severe class imbalance.
    pos_weight = torch.tensor([min(neg/pos, 20.0)], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt  = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sch  = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-5)
    use_amp = cfg.USE_AMP and device.type == "cuda"
    scaler  = GradScaler("cuda", enabled=use_amp)
    best_ap, best_state, no_improve = -1.0, None, 0

    def val_ap():
        probs = safe_inference_lstm(model, x_d_va, x_h_va, x_l_va, groups_va, device, use_amp)
        return average_precision_score(y_va_np, probs) if len(np.unique(y_va_np)) == 2 else 0.0

    for ep in range(1, cfg.MAX_EPOCHS+1):
        model.train(); opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            loss = crit(model(x_d_tr, x_h_tr, x_l_tr, groups_tr), y_tr_t)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step(ep+1e-8)
        if ep % cfg.EVAL_EVERY == 0:
            ap = val_ap()
            if ap > best_ap+1e-6: best_ap, best_state, no_improve = ap, copy.deepcopy(model.state_dict()), 0
            else: no_improve += 1
            if no_improve >= cfg.PATIENCE_CHECKS:
                if verbose: print(f"  Early stop at epoch {ep}"); break
    if best_state is not None: model.load_state_dict(best_state)
    return model


# ============================================================================
#  FOLD TRAINING  (no graph_strategy parameter, no edge construction)
# ============================================================================
def train_one_fold(df_tr, df_va, cfg, fold_idx=0, device=None):
    if device is None: device = get_device()
    df_tr, df_va, user_avg_map, global_avg = add_user_spending_features(df_tr, df_va)
    prep = FoldPreprocessor(cfg); prep.fit(df_tr)
    df_tr_raw, df_va_raw = df_tr.copy(), df_va.copy()
    X_d_tr, X_h_tr, X_l_tr, y_tr_np = prep.transform(df_tr)
    X_d_va, X_h_va, X_l_va, y_va_np = prep.transform(df_va)

    groups_tr = _build_seq_indices(df_tr_raw, cfg)
    groups_va = _build_seq_indices(df_va_raw, cfg)
    del df_tr_raw, df_va_raw; gc.collect()

    x_d_tr = torch.tensor(X_d_tr, dtype=torch.float32).to(device)
    x_h_tr = torch.tensor(X_h_tr, dtype=torch.long).to(device)
    x_l_tr = torch.tensor(X_l_tr, dtype=torch.long).to(device)
    y_tr_t  = torch.tensor(y_tr_np, dtype=torch.float32).to(device)
    x_d_va = torch.tensor(X_d_va, dtype=torch.float32).to(device)
    x_h_va = torch.tensor(X_h_va, dtype=torch.long).to(device)
    x_l_va = torch.tensor(X_l_va, dtype=torch.long).to(device)
    del X_d_tr, X_h_tr, X_l_tr, X_d_va, X_h_va, X_l_va; gc.collect()

    verbose = (fold_idx == 0)
    model = LSTMOnlyFraudModel(
        dense_in_dim=x_d_tr.shape[1],
        high_card_cols=prep.high_card_cols, low_card_cols=prep.low_card_cols,
        factor_cardinalities=prep.factor_cardinalities,
        hidden=cfg.HIDDEN_DIM, heads=cfg.HEADS,
        dropout=cfg.DROPOUT,
        high_card_emb_dim=cfg.HIGH_CARD_EMB_DIM, low_card_emb_dim=cfg.LOW_CARD_EMB_DIM,
    ).to(device)

    use_amp = cfg.USE_AMP and device.type == "cuda"
    model = train_loop_lstm(model, x_d_tr, x_h_tr, x_l_tr, y_tr_t, groups_tr,
                            x_d_va, x_h_va, x_l_va, y_va_np, groups_va, cfg, device, verbose)
    model.eval()
    p_tr = safe_inference_lstm(model, x_d_tr, x_h_tr, x_l_tr, groups_tr, device, use_amp)
    p_va = safe_inference_lstm(model, x_d_va, x_h_va, x_l_va, groups_va, device, use_amp)
    thr  = choose_threshold(y_va_np, p_va, cfg)
    del x_d_tr, x_h_tr, x_l_tr, y_tr_t, x_d_va, x_h_va, x_l_va; cleanup()
    return {"model": model.cpu(), "preprocessor": prep, "thr": thr,
             "user_avg_map": user_avg_map, "global_avg": global_avg,
             "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
             "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr)}


# ============================================================================
#  TEST ENSEMBLE
# ============================================================================
def run_test_ensemble_lstm(all_out, df_test, cfg, device):
    test_probs = []
    use_amp = cfg.USE_AMP and device.type == "cuda"
    for fold_i, out in enumerate(all_out):
        print(f"  Test fold {fold_i+1}/{len(all_out)}...")
        prep = out["preprocessor"]
        df_te = df_test.copy()
        df_te["user_avg_amount"]     = df_te["User"].map(out["user_avg_map"]).fillna(out["global_avg"]).astype("float32")
        denom = df_te["user_avg_amount"].replace(0, 1.0)
        df_te["amount_over_user_avg"]  = (df_te["Amount"]/denom).astype("float32")
        df_te["amount_minus_user_avg"] = (df_te["Amount"]-df_te["user_avg_amount"]).astype("float32")
        X_d, X_h, X_l, y_te = prep.transform(df_te)
        groups_te = _build_seq_indices(df_test, cfg)
        x_d = torch.tensor(X_d, dtype=torch.float32).to(device)
        x_h = torch.tensor(X_h, dtype=torch.long).to(device)
        x_l = torch.tensor(X_l, dtype=torch.long).to(device)
        model = out["model"].to(device)
        probs = safe_inference_lstm(model, x_d, x_h, x_l, groups_te, device, use_amp)
        test_probs.append(probs); out["model"] = model.cpu()
        del x_d, x_h, x_l; cleanup()
    p_ens = np.mean(test_probs, axis=0)
    thr_global = np.mean([o["thr"] for o in all_out])
    return p_ens, y_te, thr_global


# ============================================================================
#  PIPELINE  (runs ONCE per dataset -- no graph_strategy loop)
# ============================================================================
def run_pipeline(df, cfg):
    set_seed(cfg.SEED); device = get_device()
    print(f"\n{'#'*60}\n# LSTM-only (no graph)\n{'#'*60}")
    df_dev, df_test = split_groups_holdout(
        df, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    folds = build_group_stratified_folds(
        df_dev, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.N_SPLITS, cfg.STRATIFY_BINS, cfg.SEED)
    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"  Fold {i}/{cfg.N_SPLITS}")
        out = train_one_fold(df_tr, df_va, cfg, i-1, device)
        print_metrics("TRAIN", out["m_tr"]); print_metrics("VAL", out["m_va"])
        all_out.append(out); cleanup()
    p_ens, y_test, thr = run_test_ensemble_lstm(all_out, df_test, cfg, device)
    m_test = eval_from_probs(y_test, p_ens, thr); print_metrics("TEST", m_test)
    result = {"all_out": all_out, "df_dev": df_dev, "df_test": df_test,
               "thr_global": thr, "test_metrics": m_test,
               "p_test_ens": p_ens, "y_test": y_test,
               "graph_strategy": "no_topology", "model_arch": MODEL_ARCH}
    save_results_plots(result, cfg)
    return result


def run_all_strategies(df, cfg=None):
    """Named run_all_strategies for drop-in compatibility with the
    notebook's all_results_ibm.update(...) pattern used for every other
    architecture file, even though this model only produces ONE result
    (topology is not a variable it responds to)."""
    if cfg is None: cfg = IBMFraudConfig()
    result = run_pipeline(df, cfg)
    results = {f"no_topology_{MODEL_ARCH}": result}
    print_final_comparison(results)
    save_strategy_comparison(results, cfg)
    save_results_csv(results, cfg)
    return results


if __name__ == "__main__":
    cfg = IBMFraudConfig()
    df  = load_and_preprocess(path="/path/to/ibm_transactions.parquet", cfg=cfg)
    results = run_all_strategies(df, cfg)
