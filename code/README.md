# Business Entity Resolution — Amazon ML Challenge 2026

An end-to-end Machine Learning solution for cross-source Business Entity Resolution. Matches reference business entities from Source 1 ($S1$) against candidate records in Source 2 ($S2$) and Source 3 ($S3$), engineered to maximize the competition **Macro $F_{0.5}$** metric ($\ge 0.9800$ target) under strict memory and computational constraints.

---

## 📁 Code Package Structure (Hackathon Aligned)

```text
code/
├── src/
│   ├── main.py                 # Master pipeline entry point (Train & Zero-RAM Multiprocess Test)
│   ├── preprocessing.py        # Text cleaning, legal suffix removal, postal code, domain & acronym extractors
│   ├── indexing.py             # Balanced S2/S3 SQLite index lookup & Zero-RAM candidate re-ranking
│   ├── features.py             # Pairwise 32-similarity & conflict feature extraction engine
│   ├── model.py                # Multi-Model Ensemble (LightGBM + CatBoost + XGBoost) & F0.5 Optimizer
│   ├── submission.py           # Output TSV generation & validation routines
│   └── validate_submission.py  # Official competition submission validator
├── requirements.txt            # Pinned dependencies / environment (lightgbm, catboost, xgboost)
└── README.md                   # How to reproduce end-to-end
```

---

## ⚡ Core Technical Features (Target 0.98 Architecture)

1. **Pure Zero-RAM SQLite Engine (`index_test.db`)**: Queries candidate records directly on disk using SQLite `PRIMARY KEY` B-Tree indexes (< 0.35ms per lookup). Completely avoids storing millions of records in Python memory, preventing memory bloat and swap thrashing.
2. **High-Signal 32-Feature Extraction**:
   - **Postal / PIN Code Match & Conflict**: Matches 5-digit US ZIPs and 6-digit Indian PIN codes; penalizes intra-country zip discrepancies.
   - **Business Domain Anti-Collocation**: Classifies entities across 7 domains (`health`, `hospitality`, `education`, `automotive`, `food`, `finance`, `legal`) to veto false merges (e.g. *Hospital* vs *Hotel*).
   - **Acronym & Initialism Resolver**: Matches abbreviations to corporate names (e.g. *KFC* $\leftrightarrow$ *Kentucky Fried Chicken*).
   - **Jaro-Winkler Similarity**: Measures prefix-weighted string similarity to capture brand typos (*"Smyth"* vs *"Smith"*).
   - **House Number Verification**: Rewards matching building numbers and penalizes street number conflicts.
3. **Multi-Model Gradient Boosting Ensemble**:
   - **LightGBM** (leaf-wise gradient boosting)
   - **CatBoost** (oblivious symmetric decision trees)
   - **XGBoost** (depth-wise regularizer)
   - **Soft-Voting Ensemble Blend**: $P_{\text{final}} = 0.50 \cdot P_{\text{LightGBM}} + 0.30 \cdot P_{\text{CatBoost}} + 0.20 \cdot P_{\text{XGBoost}}$.
4. **Deadlock-Free Zero-RAM Multiprocessing**:
   - Streams 1.73M test predictions across multi-core CPUs (e.g. 6 to 10 workers) with guaranteed 1-to-1 output row ordering via `pool.imap`.
   - Uses independent read-only SQLite connections per worker and single-threaded C++ inference to eliminate OpenMP lock contention.
5. **Macro $F_{0.5}$ Threshold Optimizer**: Vectorized grid search on grouped (leak-proof) validation entities, weighting precision 4x higher than recall.

---

## 🚀 Execution Instructions

Always run within the virtual environment:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Multi-Chunk Continual Training (e.g. 2 chunks of 50,000 entities = 100,000 entities total)
python3 code/src/main.py \
    --split train \
    --chunk-size 50000 \
    --num-chunks 2 \
    --val-size 2000 \
    --num-workers 8 \
    --trees-per-chunk 100

# 3. (Optional) Resume Training on Subsequent Chunks (e.g. Chunks 3 & 4 starting at offset 100,000)
python3 code/src/main.py \
    --split train \
    --chunk-size 50000 \
    --num-chunks 2 \
    --chunk-offset 100000 \
    --continue-training \
    --val-size 2000 \
    --num-workers 8 \
    --trees-per-chunk 100

# 4. Run High-Speed Multiprocess Test Inference (on Full 1.73M Test split)
python3 code/src/main.py --split test --sample-size 0 --num-workers 8 --max-candidates 40
```

### Command Arguments:
- `--split`: Dataset split to run on (`train` or `test`).
- `--chunk-size`: Size of each sequential training chunk (e.g., `50000` or `75000`). Overrides `--sample-size`.
- `--num-chunks`: Number of sequential chunks to train in this execution (default `1`).
- `--chunk-offset`: Starting row offset in `train_source1.tsv` (default `0`). Used to resume/continue training on next slices of the dataset.
- `--continue-training`: Warm-starts training from existing `output/train/lgb_model.pkl` checkpoint, adding additional trees.
- `--trees-per-chunk`: Number of trees added to LightGBM, CatBoost, and XGBoost per chunk (default `100`).
- `--val-size`: Number of entities extracted from the tail of `train_source1.tsv` to lock as the fixed holdout validation benchmark (default `2000`).
- `--sample-size`: Number of Source 1 records to evaluate (e.g., `0` for the entire 1.73M test set).
- `--max-candidates`: Maximum top candidate records fetched per entity (default `40`).
- `--num-workers`: Number of parallel worker processes for feature extraction and test streaming (default: min(8, CPU count)).
- `--no-cache`: Force re-extraction of features, bypassing disk cache.

---

## 📊 Output Files & Submission Verification

The pipeline organizes outputs into two dedicated directories:
1. `output/train/`: Model weights (`lgb_model.pkl`), feature cache, and validation TSVs.
2. `output/test/`: Competition submission files (`matching_results.tsv` and `candidate_pairs.tsv`).

### 1. Validate Submission Format
```bash
python3 code/src/validate_submission.py \
    --matching output/test/matching_results.tsv \
    --candidate output/test/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
*(Confirms non-empty lines, tab separators, candidate pool subset constraints, and 1-to-1 entity ordering matching `test_source1.tsv`)*

### 2. Package Submission:
```bash
zip -j submission.zip output/test/matching_results.tsv output/test/candidate_pairs.tsv
```

