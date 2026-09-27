# Business Entity Resolution — Amazon ML Challenge 2026

An industrial-grade, end-to-end Machine Learning pipeline for large-scale cross-source Business Entity Resolution. Resolves reference business entities from Source 1 ($S_1$) against candidate records across Source 2 ($S_2$) and Source 3 ($S_3$), specifically engineered to maximize the official **Competition Macro $F_{0.5}$** metric ($\ge 0.9000+$) under strict memory, compute, and disk constraints.

---

## 📁 Repository & Submission Package Structure

```text
code/
├── src/
│   ├── main.py                 # Master pipeline entry point (Multi-Chunk Training & Zero-RAM Test Streaming)
│   ├── preprocessing.py        # C-speed text cleaning, legal suffixes, postal code, domain & acronym extractors
│   ├── indexing.py             # Balanced S2/S3 SQLite index lookup, conjunction queries & Zero-RAM candidate re-ranking
│   ├── features.py             # 32-dimensional pairwise similarity, veto, and conflict feature extraction engine
│   ├── model.py                # Multi-Model Ensemble (LightGBM + CatBoost + XGBoost) & Macro F0.5 Optimizer
│   ├── submission.py           # Output TSV generation & competition validation routines
│   ├── validate_submission.py  # Official competition submission validator (provided by hackathon)
│   └── sagemaker_pipeline.ipynb# Full interactive execution notebook for AWS SageMaker
├── requirements.txt            # Pinned dependencies (lightgbm, catboost, xgboost, scikit-learn, pandas)
└── README.md                   # Complete execution and reproduction guide
```

---

## ☁️ AWS SageMaker Reproduction Guide

This pipeline is optimized to run on **AWS SageMaker**. Follow these exact instructions to reproduce the pipeline from scratch.

### 1. Instance Selection & Sizing
- **Recommended Instance:** `ml.c5.18xlarge` (72 vCPUs, 144 GB RAM, compute-optimized).
- **Alternative Instances:** `ml.m5.16xlarge` (64 vCPUs, 256 GB RAM) or `ml.c5.9xlarge` (36 vCPUs, 72 GB RAM).

### 2. EBS Volume Sizing (Crucial — Avoid Disk-Full Failures)
Handling 10.1 million candidate records and generating search B-Trees requires sufficient local block storage.
- **Required Storage:** Allocate at least **150 GB to 200 GB EBS GP3 volume** (minimum 3,000 IOPS, 125 MB/s throughput).
- **Disk Allocation Breakdown:**
  | Component | Size on Disk | Description |
  | :--- | :--- | :--- |
  | Raw TSV Datasets | ~15 GB | 7 uncompressed TSVs across train and test |
  | `index.db` (Train Index) | ~14 GB | SQLite B-Trees indexing 63.3M token occurrences |
  | `index_test.db` (Test Index) | ~12 GB | SQLite B-Trees indexing test candidate pool |
  | Feature Caches & Models | ~10 GB | Extracted pair features & ensemble checkpoints |
  | System, Python & Pip Cache | ~10 GB | Virtual environment and libraries |
  | **Recommended Headroom** | **150 - 200 GB** | Safe operating buffer |

- **How to Create / Expand EBS Space on AWS SageMaker:**
  - **New Instance:** When creating an instance in the SageMaker Console, under **Notebook instance settings** $\rightarrow$ **Additional configuration**, set **Volume size in GB** to `200`.
  - **Existing Instance:** 
    1. Stop the Notebook Instance in SageMaker Console.
    2. Click **Update settings**.
    3. Increase **Volume size in GB** to `200`.
    4. Save and Start the instance.
    5. In the terminal, verify available storage with `df -h /`.

---

## 🚀 Step-by-Step Execution Workflows

### Workflow A: Interactive Notebook (`sagemaker_pipeline.ipynb`)
Open `sagemaker_pipeline.ipynb` in SageMaker JupyterLab / Studio and run cells sequentially:
- **Cell 1:** Virtual environment (`.venv`) creation & dependency installation.
- **Cell 2:** Git synchronization (`git pull origin main`).
- **Cell 3:** Verification of dataset files in `dataset/train/` and `dataset/test/`.
- **Cell 4:** Multi-chunk continual training on `train_source1.tsv`.
- **Cell 5a:** Standalone validation audit on 5,000 holdout entities (`--eval-only`).
- **Cell 5b:** Confusion matrix visualization & official Macro $F_{0.5}$ heatmap.
- **Cell 6:** Full test inference on 1.73M test entities (`--split test`).
- **Cell 7:** Verification against official competition validator (`validate_submission.py`).
- **Cell 8:** Preview top predictions in `matching_results.tsv`.
- **Cell 9:** Automated packaging into `unwanted_submission.zip`.

---

### Workflow B: Command-Line (Terminal)

#### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r code/requirements.txt
```

#### 2. Multi-Chunk Continual Training
Train sequential chunks of 40,000 entities with 36 parallel workers:
```bash
python3 code/src/main.py \
    --split train \
    --num-chunks 4 \
    --chunk-size 40000 \
    --chunk-offset 0 \
    --max-candidates 180 \
    --val-size 5000 \
    --num-workers 36 \
    --trees-per-chunk 100
```
*Note: If interrupted, resume training from the last saved checkpoint with `--continue-training` and set `--chunk-offset` to the desired row index.*

#### 3. Standalone Validation Audit (Fast Benchmark)
Evaluate existing model checkpoint against the 5,000 holdout validation cohort:
```bash
python3 code/src/main.py \
    --split train \
    --val-size 5000 \
    --max-candidates 180 \
    --num-workers 36 \
    --no-cache \
    --eval-only
```

#### 4. Full Test Inference (1.73M Entities, Zero-RAM Streaming)
```bash
python3 code/src/main.py \
    --split test \
    --sample-size 0 \
    --max-candidates 180 \
    --num-workers 36 \
    --output-dir output/
```
Outputs are automatically written to `output/matching_results.tsv` and `output/candidate_pairs.tsv` (and mirrored to `unwanted_submission/output/`).

#### 5. Validate Outputs Against Official Competition Rules
```bash
python3 code/src/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

#### 6. Package Final Submission Archive
```bash
python3 package_submission.py
```
This builds `unwanted_submission.zip` matching the exact competition directory hierarchy:
```text
unwanted_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── requirements.txt
│       └── README.md
└── Documentation_template.md
```

---

## ⚡ Core Technical Innovations

1. **Multi-Pass Conjunction Blocking (Candidate Recall: 88.25%)**:
   - **Pass 1:** 2-Term Name Conjunction (`Name1 INTERSECT Name2 LIMIT 300`)
   - **Pass 2:** Name INTERSECT House Number (`LIMIT 200`)
   - **Pass 3:** Name INTERSECT Distinctive Address Word (`LIMIT 200`)
   - **Pass 3b:** 2-Term Address Conjunction (`Addr1 INTERSECT Addr2 LIMIT 150`) — catches businesses where name is blank/abbreviated in S2/S3.
   - **Pass 4 & 5:** Balanced forward (S2) and reverse rowid (S3) scans.
   - **Pass 6 & 7:** House number and postal code blocking queries.
   - **Lexical Re-Ranking:** Selects top $k=180$ candidates directly on disk.

2. **32-Dimensional Pairwise Feature Extractor**:
   - **Postal / PIN Code Match & Conflict**: 5-digit US ZIPs & 6-digit Indian PIN codes with conflict veto.
   - **Commercial Domain Anti-Collocation**: 7 domain categories (`health`, `hospitality`, `education`, `automotive`, `food`, `finance`, `legal`) vetoing co-located businesses.
   - **Acronym & Initialism Resolver**: Matches brand abbreviations (e.g. *KFC* $\leftrightarrow$ *Kentucky Fried Chicken*).
   - **Prefix-Weighted Metrics**: Jaro-Winkler similarity on name and address.
   - **Building / House Numbers**: House number match bonus and conflict penalty.

3. **Multi-Model Gradient Boosting Stacking**:
   - Soft-voting ensemble: **LightGBM** (50%) + **CatBoost** (30%) + **XGBoost** (20%).
   - Continual booster expansion via `init_model` across training chunks.

4. **Vectorized Official Macro $F_{0.5}$ Threshold Optimizer**:
   - Auto-tunes decision threshold (optimal $\approx 0.585$) directly maximizing the official competition metric.
   - Accurately models singleton scoring ($1.0$ for clean singletons, $0.0$ for false merges).
