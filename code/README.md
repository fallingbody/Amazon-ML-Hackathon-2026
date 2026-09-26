# Business Entity Resolution — Amazon ML Challenge 2026

An end-to-end Machine Learning solution for cross-source Business Entity Resolution. Matches reference business entities from Source 1 ($S1$) against candidate records in Source 2 ($S2$) and Source 3 ($S3$), optimized for the **Macro $F_{0.5}$** metric under strict memory constraints.

---

## 📁 Code Package Structure (Hackathon Aligned)

```text
code/
├── src/
│   ├── main.py                 # Master pipeline entry point
│   ├── preprocessing.py        # Text cleaning, legal suffix removal, token extraction
│   ├── indexing.py             # Balanced S2/S3 SQLite index lookup (Zero-RAM candidate retrieval)
│   ├── features.py             # Pairwise 25-similarity & conflict feature extraction
│   ├── model.py                # LightGBM binary classifier & Macro F0.5 threshold optimization
│   ├── submission.py           # Output TSV generation & validation routines
│   └── validate_submission.py  # Official competition submission validator
├── requirements.txt            # Pinned dependencies / environment
└── README.md                   # How to reproduce end-to-end
```

---

## ⚡ Core Technical Features

1. **Zero-RAM Disk Index (`index.db`)**: Uses an indexed SQLite B-Tree database storing candidate tokens across 10,320,219 records. Enables $O(\log N)$ lookup speed while utilizing $< 50$ MB RAM (eliminates OOM kernel crashes).
2. **Balanced Cross-Source Retrieval**: Queries Source 2 and Source 3 concurrently to eliminate Source 3 B-Tree starvation.
3. **Open-Set Country Support**: Feature formulas and country match indicators do not assume fixed geographic domains, supporting new countries present in unseen test sets.
4. **Macro $F_{0.5}$ Optimization**: Tunes probability decision thresholds specifically for the $F_{0.5}$ score with official singleton scoring, weighting precision 4x higher than recall.
5. **PyArrow Memory-Mapped Dataframes**: Loads large multi-gigabyte TSV files efficiently using PyArrow string backends.

---

## 🚀 Execution Instructions

Activate the Python virtual environment and run the pipeline runner:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run the main resolution pipeline (on Train or Test split)
python3 code/src/main.py --sample-size 50000 --max-candidates 30 --split train
```

### Command Arguments:
- `--sample-size`: Number of Source 1 records to evaluate (default `50000`, set to `0` or omit on test for full dataset run).
- `--max-candidates`: Maximum number of top candidate records fetched per entity from `index.db` (default `30`).
- `--split`: Dataset split to run on (`train` or `test`).
- `--no-cache`: Force re-extraction of features.

---

## 📊 Output Files

The pipeline organizes outputs cleanly into two dedicated directories:
1. `output/train/`: Model weights (`lgb_model.pkl`), feature cache, and validation TSVs.
2. `output/test/`: Competition submission files (`matching_results.tsv` and `candidate_pairs.tsv`) ready for leaderboard upload.
