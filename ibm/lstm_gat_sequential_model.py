""" new
Sequential LSTM->GAT hybrid fraud detector for the IBM dataset.

Drop-in sibling of gatv2_model.py / gcn_model.py / graphsage_model.py.
Reuses every leakage-safe / evaluation component from utils.py unchanged
(FoldPreprocessor, split_groups_holdout, build_group_stratified_folds,
build_graph_edges, choose_threshold, eval_from_probs, print_metrics,
save_results_*). The ONLY new pieces are:

  1. _build_seq_indices()   - per-node sequence grouping for the LSTM branch.
                               Independent of graph_strategy (phi): it uses
                               the same GROUP_KEY/sort-key logic already used
                               for intra_group edges, so swapping graph
                               strategies does not change how sequences are
                               built, and swapping this model in does not
                               change how graphs are built.
  2. LSTMGATFraudModel      - encode() is copied verbatim from GATFraudModel
                               so the two architectures see identical inputs;
                               only the encoder stack differs (LSTM -> GAT
                               vs GAT alone). hidden/heads default to the
                               SAME values as cfg.HIDDEN_DIM / cfg.HEADS so
                               capacity is comparable, not just structure.
  3. train_loop_lstm / safe_inference_lstm / train_one_fold / run_pipeline
                             - thin copies of the utils.py equivalents,
                               extended only to also pass seq_indices through
                               the forward() call. Everything else (AdamW,
                               CosineAnnealingWarmRestarts, grad clipping,
                               early stopping on val AP, pos_weight BCE,
                               5-fold ensemble at test time) is identical.

Graph strategies: multi_relation | hybrid | intra_group
Run:  python lstm_gat_sequential_model.py
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
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import average_precision_score

from config import IBMFraudConfig
from utils import (
    set_seed, get_device, cleanup,
    load_and_preprocess, add_user_spending_features,
    FoldPreprocessor, split_groups_holdout, build_group_stratified_folds,
    build_graph_edges, _make_sort_key,
    eval_from_probs, print_metrics, choose_threshold,
    save_results_plots, save_strategy_comparison,
    save_results_csv, print_final_comparison,
)

MODEL_ARCH = "lstm_gat_seq"


# ============================================================================
#  SEQUENCE CONSTRUCTION (the one genuinely new preprocessing step)
# ============================================================================
def _build_seq_indices(df_raw, cfg):
    """
    Group nodes by cfg.INTRA_GROUP_KEY (same key already used for the
    intra_group graph strategy) and sort by the same temporal sort key
    used everywhere else in this codebase (cfg.SORT_KEY_COLS).

    Oversized groups are CHUNKED into consecutive windows of
    INTRA_MAX_GROUP_SIZE, not truncated. Truncating silently dropped every
    transaction past the cap from the LSTM pass entirely -- those nodes'
    entries in `output` (see _run_lstm_over_groups) were never written, so
    they stayed at their zero-initialized value and were fed into GAT as
    literal all-zero feature vectors. For any card with more transactions
    than INTRA_MAX_GROUP_SIZE, that meant real transactions were silently
    replaced with garbage input rather than merely losing long-range
    context. Chunking loses cross-chunk temporal context at the window
    boundary but guarantees every node gets a genuine LSTM output.

    Note: this is independent of `graph_strategy`. Whichever phi built the
    edges, the LSTM sequence grouping is always this same per-card
    chronological ordering, so swapping strategies never changes how
    sequences are built, and swapping this model in never changes how
    graphs are built.
    """
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
    """
    Returns: [N, lstm_hidden*2] -- for every node, the LSTM's output at
             ITS OWN position in its card's chronological sequence (i.e.
             causal: informed by that card's history up to and including
             this transaction, not the whole card averaged into one vector).
    """
    N = x.size(0)
    out_dim = lstm.hidden_size * (2 if lstm.bidirectional else 1)
    output = torch.zeros(N, out_dim, device=device)

    if not groups:
        return output

    lengths = [len(g) for g in groups]
    max_len = max(lengths)
    batch = len(groups)

    padded = torch.zeros(batch, max_len, x.size(1), device=device)
    for i, g in enumerate(groups):
        padded[i, :len(g)] = x[torch.as_tensor(g, device=device)]

    packed = nn.utils.rnn.pack_padded_sequence(
        padded, lengths, batch_first=True, enforce_sorted=False
    )
    lstm_out, _ = lstm(packed)
    unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

    for i, g in enumerate(groups):
        idx_t = torch.as_tensor(g, device=device)
        # Under AMP, `unpacked` may be Half (fp16) while `output` was
        # allocated as float32. Basic slicing assignment (as used above for
        # `padded`) casts implicitly, but advanced/fancy indexing via a
        # LongTensor (this line) requires an exact dtype match in PyTorch,
        # so we cast explicitly to avoid "Index put requires the source and
        # destination dtypes match" under mixed precision.
        output[idx_t] = unpacked[i, :len(g)].to(output.dtype)

    return output


# ============================================================================
#  MODEL
# ============================================================================
class LSTMGATFraudModel(nn.Module):
    """
    Sequential LSTM->GAT: LSTM encodes each card's chronological transaction
    sequence; its OUTPUT (not the original features) is what GAT sees.
    encode() is identical to GATFraudModel so both architectures receive the
    same categorical/dense feature construction.
    """

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

        lstm_hidden = lstm_hidden or (hidden * heads)
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden  # unidirectional: 1x, not 2x
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)  # stabilizes LSTM output
                                                       # scale before it feeds
                                                       # GAT's attention, which
                                                       # was tuned against
                                                       # encode()'s scale, not
                                                       # an LSTM's

        self.gat1 = GATv2Conv(lstm_out_dim, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(hidden*heads, hidden, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden*heads)
        self.norm2 = nn.LayerNorm(hidden*heads)
        self.cls = nn.Sequential(
            nn.Linear(hidden*heads, 128), nn.LeakyReLU(),
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

    def forward(self, x_dense, x_high, x_low, edge_index, groups):
        x = F.dropout(self.encode(x_dense, x_high, x_low), p=self.dropout, training=self.training)
        x = self.lstm_norm(_run_lstm_over_groups(self.lstm, x, groups, x.device))
        h = F.leaky_relu(self.norm1(self.gat1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm2(self.gat2(h, edge_index)))
        return self.cls(h).view(-1)


# ============================================================================
#  SAFE INFERENCE / TRAIN LOOP  (copies of utils.py versions + seq_indices)
# ============================================================================
@torch.no_grad()
def safe_inference_lstm(model, x_d, x_h, x_l, edge_index, groups, device, use_amp=True):
    model.eval()
    try:
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(x_d, x_h, x_l, edge_index, groups)
        return torch.sigmoid(logits).cpu().numpy()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("  [WARN] GPU OOM — falling back to CPU")
            cleanup(); model_cpu = model.cpu()
            logits = model_cpu(x_d.cpu(), x_h.cpu(), x_l.cpu(),
                               edge_index.cpu() if edge_index is not None else None, groups)
            model.to(device); return torch.sigmoid(logits).numpy()
        raise


def train_loop_lstm(model, x_d_tr, x_h_tr, x_l_tr, y_tr_t, edge_tr, groups_tr,
                    x_d_va, x_h_va, x_l_va, y_va_np, edge_va, groups_va,
                    cfg, device, verbose=False):
    pos = max(1, int(y_tr_t.sum().item())); neg = max(1, int(len(y_tr_t)-pos))
    # Capped at 20x rather than the raw neg/pos ratio: on IBM's imbalance,
    # an uncapped pos_weight can push nearly all logits positive before the
    # (slower-to-converge, LSTM-containing) model has learned real signal --
    # this was the likely cause of the intra_group collapse (recall 0.99,
    # precision 0.09) seen before this fix.
    pos_weight = torch.tensor([min(neg/pos, 20.0)], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt  = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sch  = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-5)
    use_amp = cfg.USE_AMP and device.type == "cuda"
    scaler  = GradScaler("cuda", enabled=use_amp)
    best_ap, best_state, no_improve = -1.0, None, 0

    def val_ap():
        probs = safe_inference_lstm(model, x_d_va, x_h_va, x_l_va, edge_va, groups_va, device, use_amp)
        return average_precision_score(y_va_np, probs) if len(np.unique(y_va_np)) == 2 else 0.0

    for ep in range(1, cfg.MAX_EPOCHS+1):
        model.train(); opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            loss = crit(model(x_d_tr, x_h_tr, x_l_tr, edge_tr, groups_tr), y_tr_t)
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
#  FOLD TRAINING  (mirrors gatv2_model.py train_one_fold, + seq_indices)
# ============================================================================
def train_one_fold(df_tr, df_va, cfg, graph_strategy, fold_idx=0, device=None):
    if device is None: device = get_device()
    df_tr, df_va, user_avg_map, global_avg = add_user_spending_features(df_tr, df_va)
    prep = FoldPreprocessor(cfg); prep.fit(df_tr)
    df_tr_raw, df_va_raw = df_tr.copy(), df_va.copy()
    X_d_tr, X_h_tr, X_l_tr, y_tr_np = prep.transform(df_tr)
    X_d_va, X_h_va, X_l_va, y_va_np = prep.transform(df_va)

    verbose = (fold_idx == 0)
    edge_tr = build_graph_edges(graph_strategy, df_tr_raw, X_d_tr, X_h_tr, X_l_tr, cfg, verbose)
    edge_va = build_graph_edges(graph_strategy, df_va_raw, X_d_va, X_h_va, X_l_va, cfg, verbose)

    groups_tr = _build_seq_indices(df_tr_raw, cfg)
    groups_va = _build_seq_indices(df_va_raw, cfg)
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

    model = LSTMGATFraudModel(
        dense_in_dim=x_d_tr.shape[1],
        high_card_cols=prep.high_card_cols, low_card_cols=prep.low_card_cols,
        factor_cardinalities=prep.factor_cardinalities,
        hidden=cfg.HIDDEN_DIM, heads=cfg.HEADS,
        dropout=cfg.DROPOUT,
        high_card_emb_dim=cfg.HIGH_CARD_EMB_DIM, low_card_emb_dim=cfg.LOW_CARD_EMB_DIM,
    ).to(device)

    use_amp = cfg.USE_AMP and device.type == "cuda"
    model = train_loop_lstm(model, x_d_tr, x_h_tr, x_l_tr, y_tr_t, edge_tr, groups_tr,
                            x_d_va, x_h_va, x_l_va, y_va_np, edge_va, groups_va, cfg, device, verbose)
    model.eval()
    p_tr = safe_inference_lstm(model, x_d_tr, x_h_tr, x_l_tr, edge_tr, groups_tr, device, use_amp)
    p_va = safe_inference_lstm(model, x_d_va, x_h_va, x_l_va, edge_va, groups_va, device, use_amp)
    thr  = choose_threshold(y_va_np, p_va, cfg)
    del x_d_tr, x_h_tr, x_l_tr, y_tr_t, edge_tr, x_d_va, x_h_va, x_l_va, edge_va; cleanup()
    return {"model": model.cpu(), "preprocessor": prep, "thr": thr,
             "user_avg_map": user_avg_map, "global_avg": global_avg,
             "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
             "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr)}


# ============================================================================
#  TEST ENSEMBLE  (mirrors utils.run_test_ensemble, + seq_indices)
# ============================================================================
def run_test_ensemble_lstm(all_out, df_test, cfg, graph_strategy, device):
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
        edge_te = build_graph_edges(graph_strategy, df_test, X_d, X_h, X_l, cfg).to(device)
        groups_te = _build_seq_indices(df_test, cfg)
        x_d = torch.tensor(X_d, dtype=torch.float32).to(device)
        x_h = torch.tensor(X_h, dtype=torch.long).to(device)
        x_l = torch.tensor(X_l, dtype=torch.long).to(device)
        model = out["model"].to(device)
        probs = safe_inference_lstm(model, x_d, x_h, x_l, edge_te, groups_te, device, use_amp)
        test_probs.append(probs); out["model"] = model.cpu()
        del x_d, x_h, x_l, edge_te; cleanup()
    p_ens = np.mean(test_probs, axis=0)
    thr_global = np.mean([o["thr"] for o in all_out])
    return p_ens, y_te, thr_global


# ============================================================================
#  PIPELINE  (mirrors gatv2_model.py run_pipeline / run_all_strategies)
# ============================================================================
def run_pipeline(df, cfg, graph_strategy="multi_relation"):
    set_seed(cfg.SEED); device = get_device()
    print(f"\n{'#'*60}\n# LSTM->GAT | {graph_strategy}\n{'#'*60}")
    df_dev, df_test = split_groups_holdout(
        df, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    folds = build_group_stratified_folds(
        df_dev, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.N_SPLITS, cfg.STRATIFY_BINS, cfg.SEED)
    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"  Fold {i}/{cfg.N_SPLITS}")
        out = train_one_fold(df_tr, df_va, cfg, graph_strategy, i-1, device)
        print_metrics("TRAIN", out["m_tr"]); print_metrics("VAL", out["m_va"])
        all_out.append(out); cleanup()
    p_ens, y_test, thr = run_test_ensemble_lstm(all_out, df_test, cfg, graph_strategy, device) 
    m_test = eval_from_probs(y_test, p_ens, thr); print_metrics("TEST", m_test)
    result = {"all_out": all_out, "df_dev": df_dev, "df_test": df_test,
               "thr_global": thr, "test_metrics": m_test,
               "p_test_ens": p_ens, "y_test": y_test,
               "graph_strategy": graph_strategy, "model_arch": MODEL_ARCH}
    save_results_plots(result, cfg)
    return result


def run_all_strategies(df, cfg=None):
    if cfg is None: cfg = IBMFraudConfig()
    results = {}
    for strat in ["multi_relation", "hybrid", "intra_group"]:
        results[f"{strat}_{MODEL_ARCH}"] = run_pipeline(df, cfg, strat); cleanup()
    print_final_comparison(results)
    save_strategy_comparison(results, cfg)
    save_results_csv(results, cfg)
    return results


if __name__ == "__main__":
    cfg = IBMFraudConfig()
    df  = load_and_preprocess(path="/path/to/ibm_transactions.parquet", cfg=cfg)
    results = run_all_strategies(df, cfg)
