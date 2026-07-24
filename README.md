# Hybrid LSTM+GAT Fraud Detection

Extends the fixed-architecture, variable-topology framework from
*Isolating Graph Topology from Model Architecture in GNN-Based Fraud
Detection* (Amiri & Jaf, University of Sunderland, 2026) by holding
**topology fixed** and varying the **architecture** instead — the inverse
experiment, testing whether the paper's topology ranking
(`multi_relation` > `hybrid` > `intra_group`) survives the swap.

This repo is self-contained: `ibm/config.py`, `ibm/utils.py`, and
`ibm/gatv2_model.py` are copied from the original project so every file
here runs independently, without needing the original repository present
alongside it. Full credit for the underlying framework, graph-construction
strategies, and leakage-safe evaluation protocol belongs to the original
paper and codebase.

---

## Project Structure

```
hybrid-gnn-lstm-fraud/
├── ibm/
│   ├── config.py                         # unmodified, from the original project
│   ├── utils.py                          # unmodified, from the original project
│   ├── gatv2_model.py                    # unmodified, from the original project
│   ├── lstm_gat_sequential_model.py      # NEW — Sequential LSTM→GAT hybrid
│   ├── lstm_gat_parallel_model.py        # NEW — Parallel LSTM‖GAT hybrid
│   └── account_gat_homogeneous_model.py  # NEW — account-level (not transaction-level) GAT
│
├── notebooks/
│   └── run_all_colab.ipynb               # runs all 3 + GATv2 baseline end-to-end
│
├── requirements.txt
└── README.md
```

---

## The Three Extensions

| File | Architecture | Graph strategies | Node granularity |
|---|---|---|---|
| `lstm_gat_sequential_model.py` | LSTM encodes each card's chronological transaction sequence causally (per-node, not broadcast); GAT then reasons over the LSTM output using the same `edge_index` as the active strategy | `multi_relation`, `hybrid`, `intra_group` | Transaction |
| `lstm_gat_parallel_model.py` | LSTM and GAT read the same `encode()` output independently; merged via cross-attention (GAT features query LSTM features) before classification, instead of a pipeline | `multi_relation`, `hybrid`, `intra_group` | Transaction |
| `account_gat_homogeneous_model.py` | Reuses `GATFraudModel` **unchanged** — only the topology and node granularity change | `similarity`, `shared_merchant`, `combined` | Account (aggregated) |

All three run under the same leakage-safe protocol as the original
framework: group-aware holdout split, group-stratified 5-fold
cross-validation, train-only feature fitting, pooled-validation threshold
tuning, and 5-fold ensemble averaging at test time. Nothing about the
evaluation methodology changes — only the model (files 1–2) or the
topology/granularity (file 3) does.

`account_gat_homogeneous_model.py` collapses each account's full
transaction history into a single node (`build_account_dataframe`) and
constructs a new account-to-account graph (`build_account_graph_edges`)
instead of the transaction-level graph builder — everything else (model,
training loop, evaluation) is imported from `gatv2_model.py` / `utils.py`
without modification, since a pure-GAT model needs no new architecture
code, only a different topology.

---

## Setup

```bash
git clone https://github.com/<your-username>/hybrid-gnn-lstm-fraud.git
cd hybrid-gnn-lstm-fraud
pip install -r requirements.txt
```

You'll also need the IBM Credit Card Transactions dataset
(`reduced_dataset.parquet` or equivalent) — this repo does not redistribute
the data itself.

---

## Running

### Locally / on a server

```bash
cd ibm
python lstm_gat_sequential_model.py       # edit the __main__ block's data path first
python lstm_gat_parallel_model.py
python account_gat_homogeneous_model.py
```

Each script trains all of its graph strategies in one run and saves plots
+ CSV summaries to `../outcomes/`.

### Google Colab

Open `notebooks/run_all_colab.ipynb`, set `REPO_URL` and `IBM_DATA_PATH`
in Cell 2, and run top to bottom. It runs the GATv2 baseline plus all
three extensions and produces one combined comparison table.

---

## Results Format

Every run appends a row to `outcomes/ibm_results_summary.csv` with F1,
precision, recall, ROC-AUC, Average Precision, log-loss, and Brier score,
plus diagnostic plots (confusion matrix, ROC curve, PR curve, threshold
tuning) per architecture × strategy combination.

---

## Attribution

- Original framework, paper, graph-construction strategies (`multi_relation`,
  `hybrid`, `intra_group`), and leakage-safe evaluation protocol:
  Roya Amiri & Sardar Jaf, University of Sunderland, 2026.
- LSTM+GAT hybrid architectures and account-level extension: this repo.
