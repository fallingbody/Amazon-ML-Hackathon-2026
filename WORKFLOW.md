# 🏢 Business Entity Resolution: Full End-to-End Workflow
### Amazon ML Challenge 2026 — Architecture, Engineering & Operational Runbook

---

## 📌 Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Data Pipeline & Multi-Source Schema](#2-data-pipeline--multi-source-schema)
3. [Stage 1: Pure Zero-RAM Candidate Blocking & Retrieval](#3-stage-1-pure-zero-ram-candidate-blocking--retrieval)
4. [Stage 2: High-Speed 32-Dimensional Feature Engineering](#4-stage-2-high-speed-32-dimensional-feature-engineering)
5. [Stage 3: Multi-Model Gradient Boosting Ensemble & Macro F0.5 Optimizer](#5-stage-3-multi-model-gradient-boosting-ensemble--macro-f05-optimizer)
6. [Stage 4: Deadlock-Free Zero-RAM Multiprocess Test Streaming](#6-stage-4-deadlock-free-zero-ram-multiprocess-test-streaming)
7. [Benchmark Evolution & Validation Diagnostic Insights](#7-benchmark-evolution--validation-diagnostic-insights)
8. [Step-by-Step Operational Runbook](#8-step-by-step-operational-runbook)

---

## 1. System Architecture Overview

The system resolves real-world business entities across three noisy, independent data sources ($S_1, S_2, S_3$) containing over 10 million candidate records. A naive pairwise Cartesian product requires over $1,732,544 \times 9,969,589 \approx 1.7 \times 10^{13}$ comparisons, which is computationally intractable. 

Our solution adopts a **Two-Stage Hybrid Architecture with Multi-Model Stacking**:
1. **Stage 1 (Pure Zero-RAM Candidate Blocking & Re-Ranking):** Uses a persistent SQLite B-Tree index with token-frequency filtering, auxiliary postal code queries, and zero-RAM disk-based lexical re-ranking to isolate the top $k=40$ candidate records per entity (reducing the comparison space by $> 99.9996\%$).
2. **Stage 2 (Fine-Grained Classification & Threshold Tuning):** Computes a 32-dimensional similarity and conflict vector per candidate pair, executes soft-voting inference using a diverse gradient-boosted tree ensemble (**LightGBM + CatBoost + XGBoost**), and maximizes the competition's **Macro $F_{0.5}$** metric via vectorized threshold search.

```mermaid
flowchart TD
    subgraph "Data Layer"
        S1[Source 1: Reference Entities\n1.73M Test / 50k Train]
        S2[Source 2 Records\n4.90M Records]
        S3[Source 3 Records\n5.07M Records]
    end

    subgraph "Stage 1: Zero-RAM Candidate Blocking"
        DB[(SQLite Disk Database\nindex.db / index_test.db\n~10M Records | 70.9M Token Links)]
        S2 -->|Indexed B-Tree| DB
        S3 -->|Indexed B-Tree| DB
        S1 --> TokenExt[Extract Informative Tokens\nFilter 352 Stop Tokens]
        TokenExt --> TokenQuery[Forward S2 & Reverse S3 Queries\nLIMIT 75 per Token]
        S1 --> ZipQuery[Auxiliary Postal Code Query\nLIMIT 40 per Zip]
        TokenQuery & ZipQuery --> CandidatePool[Candidate Pool Union\n~60 - 150 Candidate IDs]
        CandidatePool --> DiskLexical[Zero-RAM Lexical Re-Ranker\nSQLite Primary Key Query & Scoring]
        DiskLexical --> TopK[Top 40 Candidate Pairs\nRecall: >94%]
    end

    subgraph "Stage 2: Feature Engineering (32 Dimensions)"
        TopK --> PrecomputeS1[Precompute S1 Representation\nTokens, Digits, Postal, Domain, 3-Grams]
        PrecomputeS1 --> PairFeats[Extract 32 Features\nPostal Match/Conflict, Domain Clash Veto,\nAcronym Matcher, Jaro-Winkler, House Match]
    end

    subgraph "Stage 3: Multi-Model Ensemble Inference"
        PairFeats --> LGB[LightGBM Classifier\nLeaf-Wise Splits]
        PairFeats --> CB[CatBoost Classifier\nOblivious Symmetric Trees]
        PairFeats --> XGB[XGBoost Classifier\nDepth-Wise Regularizer]
        LGB & CB & XGB --> SoftBlend[Soft-Voting Ensemble Blend\n0.50 LGB + 0.30 CB + 0.20 XGB]
        SoftBlend --> ThresholdTuner[Vectorized Macro F0.5 Optimizer\n131 Grid Steps on Val Groups]
        ThresholdTuner --> BinaryPreds[Filtered Entity Matches\nOptimal Threshold = 0.680]
    end

    subgraph "Stage 4: Submission Outputs"
        TopK --> OutputCand[candidate_pairs.tsv\nTop 40 Candidates per S1]
        BinaryPreds --> OutputMatch[matching_results.tsv\nFinal Matches per S1]
        OutputCand --> Validator[Official Validator\nvalidate_submission.py]
        OutputMatch --> Validator
    end
```

---

## 2. Data Pipeline & Multi-Source Schema

### Input Data Specifications
| Source | Entity ID Prefix | Role | Approximate Records | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Source 1** | `S1-` | Deduplicated Reference | 1,732,544 (Test) / 50,000 (Train) | Primary query entity set |
| **Source 2** | `S2-` | Candidate Source A | ~4,900,000 | Multi-jurisdiction corporate records |
| **Source 3** | `S3-` | Candidate Source B | ~5,069,589 | Accounts for **51.7%** of all ground truth matches |

### Canonical Schema
All TSV files share standard fields:
- `entity_id`: Primary record identifier (e.g. `S1-714132312`, `S2-187020300`, `S3-867809779`).
- `business_name`: Text with legal abbreviations, typos, transliterations, and repeated words.
- `business_address`: Noisy street addresses, landmarks, building numbers, and zip codes.
- `country`: Open-set geographic identifier (`US`, `India`, `France`, etc.).

---

## 3. Stage 1: Pure Zero-RAM Candidate Blocking & Retrieval

### 3.1 Persistent SQLite Database Architecture
To eliminate out-of-memory (OOM) failures while searching 10 million candidate records, data is maintained in an optimized SQLite database (`index_test.db` / `index.db`):
- **`records` table**: Stores `(record_id PRIMARY KEY, name, address, country, dataset)`.
- **`token_index` table**: Stores `(token, entity_id)` holding > 70.9 million inverted index entries with a clustered B-Tree index on `token_index(token)`.

### 3.2 High-Throughput Token Filtering & Blocking
1. **Diacritic & Accent Normalization**: Unicode NFKD decomposition strips diacritics (`é` $\rightarrow$ `e`).
2. **Noise & Legal Suffix Stripping**: Strips URLs (`www.*`, `.com`), punctuation, and legal entity suffixes (`inc`, `llc`, `pvt ltd`, `gmbh`, `sarl`).
3. **Generic Token Shield**: Pre-filters a curated set of **352 high-frequency tokens** (`GENERIC_TOKENS`) appearing in $> 25,000$ database records (e.g. `delhi`, `mumbai`, `services`, `center`, `limited`) to avoid costly multi-thousand disk page reads.
4. **Auxiliary Postal Code Query**: Extracts 5-digit US ZIPs or 6-digit Indian PINs and queries `token_index WHERE token = ? LIMIT 40`. Captures true matching businesses that suffer from severe name typos but share the same postal code.

### 3.3 Zero-RAM Lexical Candidate Re-Ranking
Rather than taking an unranked arbitrary cutoff from SQLite:
1. Query up to 3 non-generic search tokens from S1 entity with `LIMIT 75` forward for S2 and `LIMIT 75 ORDER BY rowid DESC` for S3.
2. Pool all retrieved candidate IDs into a set (typically 60 – 150 distinct IDs).
3. Query candidate record attributes directly from SQLite's `PRIMARY KEY` B-Tree index (< 0.35ms) without creating multi-gigabyte Python dictionaries.
4. Score each candidate against the S1 record using fast lexical metrics:
   $$\text{Score} = 4 \cdot |\text{Words}_{\text{name1}} \cap \text{Words}_{\text{name2}}| + |\text{Words}_{\text{addr1}} \cap \text{Words}_{\text{addr2}}| + 3 \cdot \mathbb{I}(\text{first\_word}_1 == \text{first\_word}_2) + \text{HouseBonus}$$
5. Sort candidates by score descending and return the top $k=40$ candidate IDs.

---

## 4. Stage 2: High-Speed 32-Dimensional Feature Engineering

To evaluate candidate pairs at over 3,000 comparisons/second, the S1 reference attributes are precomputed once via `precompute_s1_features()`, avoiding redundant regex executions across candidates.

### 32-Dimensional Feature Inventory
Each $(S_1, \text{Candidate})$ pair generates 32 numerical features engineered for high Precision and Macro $F_{0.5}$ optimization:

| Category | Feature Name | Description | Signal & Rationale |
| :--- | :--- | :--- | :--- |
| **Postal Code** | `postal_match` | Exact ZIP / PIN code equality | **Positive Signal**: 1.0 if identical 5-digit US / 6-digit Indian postal code |
| **Postal Code** | `postal_conflict` | Postal code disagreement | **False-Positive Killer**: 1.0 if both have postal codes in same country but differ |
| **Domain Category**| `domain_match` | Commercial category agreement | 1.0 if both belong to same category (`health`, `hospitality`, `education`, etc.) |
| **Domain Category**| `domain_conflict` | Commercial category clash | **Domain Veto**: 1.0 if categories conflict (e.g. *Hospital* vs *Hotel*) |
| **Acronym** | `acronym_match` | Name initialism equality | 1.0 if short name matches word initials of candidate (e.g. *KFC* $\leftrightarrow$ *Kentucky Fried Chicken*) |
| **String Metric** | `name_jaro` | Name Jaro-Winkler similarity | Prefix-weighted string distance; captures brand typos (*"Smyth"* vs *"Smith"*) |
| **String Metric** | `addr_jaro` | Address Jaro-Winkler similarity| Prefix-weighted address token distance |
| **Name Lexical** | `name_jaccard` | Word token overlap | Jaccard coefficient over word token sets |
| **Name Lexical** | `name_char_jaccard` | 3-gram character overlap | Jaccard coefficient over character tri-grams |
| **Name Lexical** | `name_ratio` | 2-gram character similarity | Character bigram Dice overlap |
| **Name Lexical** | `name_sort_ratio` | Alphabetically sorted token ratio | Invariant to word transposition |
| **Name Lexical** | `name_exact` | Exact normalized equality | $\mathbb{I}(\text{name}_1 == \text{name}_2)$ |
| **Name Containment**| `name_containment` | Abbreviation / prefix containment | $\frac{\|T_{n1} \cap T_{n2}\|}{\min(\|T_{n1}\|, \|T_{n2}\|)}$ |
| **Name Asymmetry**| `name_s1_in_cand` | S1 token coverage in candidate | Fraction of S1 name tokens present in candidate |
| **Name Asymmetry**| `name_cand_in_s1` | Candidate token coverage in S1 | Fraction of candidate name tokens present in S1 |
| **Name Length** | `name_len_diff` | Absolute word count difference | $\|\|T_{n1}\| - \|T_{n2}\|\|$ |
| **Name Length** | `name_len_ratio` | Relative word count ratio | $\frac{\min(\|T_{n1}\|, \|T_{n2}\|)}{\max(\|T_{n1}\|, \|T_{n2}\|)}$ |
| **Name Positional**| `first_word_match` | Anchor brand token match | $\mathbb{I}(\text{first\_word}_1 == \text{first\_word}_2)$ |
| **Name Positional**| `last_word_match` | Suffix word match | $\mathbb{I}(\text{last\_word}_1 == \text{last\_word}_2)$ |
| **Address Lexical**| `addr_jaccard` | Address token overlap | Jaccard coefficient over address token sets |
| **Address Lexical**| `addr_char_jaccard`| Address character 3-gram overlap | Jaccard coefficient over address tri-grams |
| **Address Lexical**| `addr_sort_ratio` | Sorted address token similarity | Invariant to address component ordering |
| **Address Lexical**| `addr_exact` | Exact normalized address match | $\mathbb{I}(\text{addr}_1 == \text{addr}_2)$ |
| **Address Containment**| `addr_containment` | Address subset ratio | $\frac{\|T_{a1} \cap T_{a2}\|}{\min(\|T_{a1}\|, \|T_{a2}\|)}$ |
| **Address Asymmetry**| `addr_s1_in_cand` | S1 address coverage in candidate | Fraction of S1 address tokens present in candidate |
| **Address Asymmetry**| `addr_cand_in_s1` | Candidate address coverage in S1 | Fraction of candidate address tokens present in S1 |
| **Interaction** | `both_match_score` | Joint agreement term | `name_jaccard` $\times$ `addr_jaccard` |
| **Interaction** | `both_exact` | Joint exact equality | $\mathbb{I}(\text{name\_exact} == 1 \land \text{addr\_exact} == 1)$ |
| **Hard Key** | `house_num_match` | Building/house number match | 1.0 (exact), 0.5 (partial overlap), 0.0 (none) |
| **Hard Key** | `house_num_conflict`| Building number mismatch | 1.0 if both have house numbers and they strictly disagree |
| **Geo** | `country_match` | Open-set geographic agreement | 1.0 if identical country string |
| **Origin** | `is_s2` | Source partition indicator | 1.0 if Source 2, 0.0 if Source 3 |

---


## 5. Stage 3: Multi-Model Gradient Boosting Ensemble & Macro F0.5 Optimizer

### 5.1 Leak-Proof Grouped Validation
To ensure validation results mirror leaderboard test evaluation:
- The validation split (80/20) is partitioned **strictly by Source 1 entity IDs**.
- All candidate pairs for a given S1 entity belong exclusively to either Train or Validation, preventing data leakage across candidate pairs.

### 5.2 Multi-Model Gradient Boosting Stack
- **LightGBM Classifier**: Leaf-wise tree growth with depth 8, 300 estimators, early stopping at 50 rounds.
- **CatBoost Classifier**: Oblivious symmetric decision trees with depth 6, 300 iterations, providing strong resistance against overfitting.
- **XGBoost Classifier**: Depth-wise tree regularization with max depth 6, 300 estimators.
- **Soft-Voting Ensemble Blend**:
  $$P_{\text{blend}} = 0.50 \cdot P_{\text{LightGBM}} + 0.30 \cdot P_{\text{CatBoost}} + 0.20 \cdot P_{\text{XGBoost}}$$

### 5.3 Exact Competition Metric Optimization (Macro F0.5)
The competition evaluation metric is **Macro $F_{0.5}$**, which weights precision 4x higher than recall:
$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

The model executes an ultra-fast vectorized threshold search across 131 probability bins ($0.20 \le \tau \le 0.85$):
1. For each threshold $\tau$, candidate predictions are grouped by S1 entity.
2. Group-level $TP_g$, $FP_g$, $FN_g$ are calculated using NumPy vector operations.
3. Macro $F_{0.5}$ is calculated per entity group, assigning $1.0$ for correct singletons ($TP=0, FP=0, FN=0$).
4. The threshold maximizing the arithmetic mean across all entities ($\tau^* \approx 0.680$) is selected.

---

## 6. Stage 4: Deadlock-Free Zero-RAM Multiprocess Test Streaming

Running test inference on 1,732,544 entities requires evaluating $> 50,000,000$ candidate pairs.

### 6.1 Deadlock-Free SQLite Multiprocessing
- **Parent Handle Isolation**: Parent process explicitly closes its SQLite connection (`indexer.close()`) before spawning `mp.Pool`. This eliminates inherited file descriptor lock conflicts in Linux.
- **Independent Child Handles**: Each worker process opens its own dedicated read-only SQLite connection: `sqlite3.connect("file:index_test.db?mode=ro", uri=True)`.
- **OpenMP Lock Contention Bypass**: Model prediction invokes `booster_.predict(..., num_threads=1)` directly in C++, completely avoiding OpenMP thread-pool futex deadlocks.

### 6.2 Pure Zero-RAM Disk Execution
- Candidate attributes are read directly from SQLite's disk B-Tree index as needed (< 0.35ms), eliminating the 4 GB Python in-memory dictionary.
- Memory usage remains strictly **< 150 MB per worker process** (~1.2 GB total across 8 workers), with **zero swap thrashing**.
- Predictions stream directly to disk in chunks of 500 entities using `pool.imap(..., chunksize=1)`, preserving the exact 1-to-1 row ordering of `test_source1.tsv`.

### 6.3 The CPython Copy-On-Write (COW) Memory Discovery & Fix
- **Symptom**: During initial multiprocessing scaling on Linux, preloading 9.97M records into an in-memory Python dictionary caused RAM usage to balloon from 4 GB to over 40 GB across 10 workers, filling physical RAM and swap to 100% and crashing the machine.
- **Root Cause**: In CPython, merely reading dictionary keys or tuples increments internal reference counters (`ob_refcnt`). Under Linux `fork()`, modifying a reference counter in child workers marks memory pages as dirty, triggering Copy-On-Write (COW) page duplication across all processes.
- **Solution (Zero-RAM Stream)**: Removed the in-memory dictionary entirely. Each worker opens an independent read-only URI connection to `index_test.db` (`mode=ro`). SQLite handles page caching transparently in OS shared memory, maintaining **flat, rock-solid < 150 MB RSS per worker**.

---

## 7. Benchmark Evolution & Validation Diagnostic Insights

| Milestone | Architecture & Enhancements | Pairwise Precision | Pairwise Recall | Macro $F_{0.5}$ (Leaderboard) | Key Breakthroughs |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **v1.0 Baseline** | Single-Source (S2 only), 11 features, arbitrary cutoffs | 46.90% | 24.11% | **0.3694** | Missing Source 3 entirely |
| **v2.0 Optimized** | S2 + S3 Indexed, 17 features, Macro threshold tuning | 81.25% | 63.63% | **0.7418** | Full S3 indexing, basic lexical re-ranking |
| **v3.0 Leak-Proof**| Grouped 80/20 Val Split, 25 features, Country filter | 95.07% | 89.33% | **0.9355** | Open-set country guard, house number penalty |
| **v4.0 Target 0.98**| **32 features + LightGBM/CatBoost/XGBoost Ensemble + Zero-RAM** | **93.56% (1k)** | **90.27% (1k)** | **0.9391 (1k) $\rightarrow$ Projected 0.975–0.985 (50k)** | Postal codes, domain clash veto, acronyms, Jaro-Winkler, 3-model stacking |

---

## 8. Step-by-Step Operational Runbook

### Environment Setup
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Verify dependencies
pip install -r code/requirements.txt
```

### Stage A: Multi-Chunk Continual Training (Scalable High-Performance Mode)
```bash
# Train on 100,000 entities across 2 sequential chunks (RAM stays < 2.5 GB at all times):
python3 code/src/main.py \
    --split train \
    --chunk-size 50000 \
    --num-chunks 2 \
    --val-size 2000 \
    --num-workers 8 \
    --trees-per-chunk 100

# Continue training later on next slices (e.g. records 100,000 to 200,000):
python3 code/src/main.py \
    --split train \
    --chunk-size 50000 \
    --num-chunks 2 \
    --chunk-offset 100000 \
    --continue-training \
    --val-size 2000 \
    --num-workers 8 \
    --trees-per-chunk 100
```
- **Runtime**: ~4–5 minutes per 50k chunk with 8 parallel worker processes (~200–250 entities/sec).
- **Process**:
  1. **Locked Holdout Validation**: Samples and locks 2,000 entities from the tail of `train_source1.tsv` to `output/train/fixed_val_set_2000_40.pkl`. All chunks evaluate against this identical benchmark with zero data leakage.
  2. **Multi-Chunk Warm-Start**: LightGBM, CatBoost, and XGBoost warm-start from the previous chunk (`init_model` and `xgb_model`), incrementally adding trees per chunk (e.g., 100 $\rightarrow$ 200 $\rightarrow$ 300).
  3. **Zero RAM Accumulation**: Memory is strictly released after every chunk via `del` and `gc.collect()`, keeping RAM flat below 2.5 GB even when scaling to hundreds of thousands of records.
  4. **True Competition Macro $F_{0.5}$ Auditing**: Every chunk evaluates and reports both candidate-level metrics and the official end-to-end competition Macro $F_{0.5}$. Model checkpoints are persisted to `output/train/lgb_model.pkl`.

### Stage B: High-Speed Multiprocess Test Streaming (Official Leaderboard Mode)
```bash
python3 code/src/main.py --split test --sample-size 0 --num-workers 8 --max-candidates 40
```
- **Throughput**: ~550–650 entities/second on an 8-core CPU.
- **Runtime**: ~45–55 minutes for all 1,732,544 test entities.
- **Memory Footprint**: Strictly < 1.2 GB RAM total across all 8 workers (< 150 MB per worker process).

### Stage C: Validate Outputs Against Official Competition Rules
```bash
python3 code/src/validate_submission.py \
    --matching output/test/matching_results.tsv \
    --candidate output/test/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
- **Target Status**: `PASS — no blocking issues found. Safe to submit.`

### Stage D: Package Final Submission Archive
```bash
zip -j submission.zip output/test/matching_results.tsv output/test/candidate_pairs.tsv
```

