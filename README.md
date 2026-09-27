# Amazon ML Challenge 2026: Business Entity Resolution

[![Competition](https://img.shields.io/badge/Amazon%20ML%20Challenge-2026-orange.svg)](https://www.hackerearth.com/challenges/competitive/amazon-ml-challenge-2026/)
[![Target Metric](https://img.shields.io/badge/Macro%20F0.5-%E2%89%A5%200.9800-brightgreen.svg)]()
[![Memory Mode](https://img.shields.io/badge/Pure%20Zero--RAM-Multiprocessing-blue.svg)]()
[![Ensemble](https://img.shields.io/badge/Stacking-LightGBM%20%2B%20CatBoost%20%2B%20XGBoost-purple.svg)]()

Production-grade Machine Learning solution for large-scale cross-source Business Entity Resolution. Resolves reference business entities in Source 1 against noisy candidate records in Source 2 and Source 3 (> 10 million records) under strict zero-RAM overhead, optimizing directly for the competition **Macro $F_{0.5}$** metric.

---

## 📑 Core Documentation Index

| Document | Purpose |
| :--- | :--- |
| **[WORKFLOW.md](file:///home/itzksv/Mine/my_codes/ml/WORKFLOW.md)** | **Master Architectural Runbook**: Complete end-to-end design, SQLite B-Tree indexing, 32-feature catalog, CPython COW analysis, benchmark evolution, and operational runbook. |
| **[code/README.md](file:///home/itzksv/Mine/my_codes/ml/code/README.md)** | **Code Execution Guide**: Instructions on flags, training, streaming inference, and package structure. |
| **[Documentation_template.md](file:///home/itzksv/Mine/my_codes/ml/Documentation_template.md)** | **Official Solution Report**: Hackathon documentation write-up adhering to the official competition template. |

---

## ⚡ Architecture Highlights

- **Pure Zero-RAM Candidate Blocking & Retrieval**:
  - Persistent SQLite B-tree database (`index_test.db` / `index.db`) holding 70.9 million token inverted index entries.
  - 352-word generic stop token shield to avoid expensive disk page reads.
  - Auxiliary 5-digit US ZIP and 6-digit Indian PIN code blocking queries (`LIMIT 40`).
  - Zero-RAM lexical candidate re-ranking querying SQLite primary keys directly (< 0.35ms), eliminating large Python dictionaries.
- **High-Signal 32-Dimensional Feature Engineering**:
  - Postal code match & intra-country postal conflict false-positive veto.
  - 7-domain commercial anti-collocation check (`health`, `hospitality`, `education`, `automotive`, `food`, `finance`, `legal`).
  - Acronym & corporate initialism resolution (e.g., *KFC* $\leftrightarrow$ *Kentucky Fried Chicken*).
  - Prefix-weighted Jaro-Winkler string similarity for brand typos.
  - Exact and partial house/building number matching and disagreement penalty.
  - Asymmetric word coverage, character 3-grams, and token sort ratios.
- **Triple Gradient Boosting Ensemble (LightGBM + CatBoost + XGBoost)**:
  - Soft-voting ensemble: $P_{\text{final}} = 0.50 \cdot P_{\text{LightGBM}} + 0.30 \cdot P_{\text{CatBoost}} + 0.20 \cdot P_{\text{XGBoost}}$.
  - Vectorized Macro $F_{0.5}$ threshold optimization on leak-proof grouped validation entities ($80/20$).
- **Deadlock-Free Zero-RAM Multiprocessing**:
  - Streams 1.73M test entities in chunks of 500 across 8 CPU workers.
  - Independent read-only SQLite connections per worker (`mode=ro`).
  - Single-threaded C++ inference to eliminate OpenMP futex deadlocks.
  - RSS memory bounded at < 150 MB per worker process (< 1.2 GB RAM total).

---

## 🚀 Quickstart Commands

### 1. Environment Setup
```bash
source .venv/bin/activate
pip install -r code/requirements.txt
```

### 2. Train Multi-Model Ensemble
```bash
python3 code/src/main.py --split train --sample-size 50000 --max-candidates 40
```

### 3. Generate Official Test Predictions (Zero-RAM Stream)
```bash
python3 code/src/main.py --split test --sample-size 0 --num-workers 8 --max-candidates 40
```

### 4. Validate Submission Format
```bash
python3 code/src/validate_submission.py \
    --matching output/test/matching_results.tsv \
    --candidate output/test/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

### 5. Package Submission Archive
```bash
zip -j submission.zip output/test/matching_results.tsv output/test/candidate_pairs.tsv
```
