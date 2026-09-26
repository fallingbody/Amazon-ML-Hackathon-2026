# Business Entity Resolution — Amazon ML Challenge 2026

An end-to-end Machine Learning solution for cross-source Business Entity Resolution. Matches reference business entities from Source 1 ($S1$) against candidate records in Source 2 ($S2$) and Source 3 ($S3$), optimized for the **Macro $F_{0.5}$** metric under strict memory constraints.

---

## 📁 Repository Structure

```text
code/
├── src/
│   ├── preprocessing.py    # Text cleaning, legal suffix removal, token extraction
│   ├── indexing.py         # SQLite index.db lookup interface (Zero-RAM candidate retrieval)
│   ├── features.py         # Pairwise similarity feature extraction (Jaccard, Levenshtein, Phone, Email)
│   ├── model.py            # LightGBM binary classifier & Macro F0.5 threshold optimization
│   └── submission.py       # Output TSV generation & validation routines
├── main.py                 # Master pipeline runner
├── requirements.txt        # Required Python packages
└── README.md               # Pipeline documentation
```

---

## ⚡ Core Technical Features

1. **Zero-RAM Disk Index (`index.db`)**: Uses an indexed SQLite B-Tree database storing candidate tokens across 10,320,219 records. Enables $O(\log N)$ lookup speed while utilizing $< 50$ MB RAM (eliminates OOM kernel crashes).
2. **Open-Set Country Support**: Feature formulas and country match indicators do not assume fixed geographic domains, supporting new countries present in unseen test sets.
3. **Macro $F_{0.5}$ Optimization**: Tunes probability decision thresholds specifically for the $F_{0.5}$ score, weighting precision higher than recall ($F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$).
4. **PyArrow Memory-Mapped Dataframes**: Loads large multi-gigabyte TSV files efficiently using PyArrow string backends.

---

## 🚀 Execution Instructions

Activate the Python virtual environment and run the pipeline runner:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run the main resolution pipeline (on Train or Test split)
python3 code/main.py --sample-size 50000 --max-candidates 30 --split train
```

### Command Arguments:
- `--sample-size`: Number of Source 1 records to evaluate (default `50000`, set higher or omit for full dataset run).
- `--max-candidates`: Maximum number of top candidate records fetched per entity from `index.db` (default `30`).
- `--split`: Dataset split to run on (`train` or `test`).

---

## 📊 Output Files

The pipeline generates two compliance-verified files in `output/`:
1. `output/candidate_pairs.tsv`: Evaluated pairs (`source1_id\tcandidate_id`).
2. `output/matching_results.tsv`: Final predicted target entity lists (`source1_id\ttarget_id_list`).
