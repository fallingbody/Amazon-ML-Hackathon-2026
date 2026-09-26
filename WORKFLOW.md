# 🏢 Business Entity Resolution: Full End-to-End Workflow

An exhaustive architectural and operational reference for the Amazon ML Challenge 2026 Business Entity Resolution solution.

---

## 📌 Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Data Pipeline & Multi-Source Schema](#2-data-pipeline--multi-source-schema)
3. [Stage 1: Zero-RAM Candidate Blocking & Retrieval](#3-stage-1-zero-ram-candidate-blocking--retrieval)
4. [Stage 2: High-Speed Pairwise Feature Engineering](#4-stage-2-high-speed-pairwise-feature-engineering)
5. [Stage 3: LightGBM Classifier & Macro F0.5 Threshold Tuning](#5-stage-3-lightgbm-classifier--macro-f05-threshold-tuning)
6. [Stage 4: Output Generation & Submission Validation](#6-stage-4-output-generation--submission-validation)
7. [Benchmark Evolution & Diagnostic Insights](#7-benchmark-evolution--diagnostic-insights)
8. [Step-by-Step Operational Runbook](#8-step-by-step-operational-runbook)

---

## 1. System Architecture Overview

The system resolves real-world business entities across three noisy, independent data sources ($S1, S2, S3$) containing over 10 million candidate records. A naive pairwise comparison requires over $50,000 \times 10,185,603 \approx 5 \times 10^{11}$ comparisons, which is computationally intractable. 

Our solution adopts a **Two-Stage Hybrid Architecture**:
1. **Stage 1 (Blocking & Candidate Retrieval):** Uses a persistent, zero-RAM SQLite B-Tree database with rare-token candidate pooling and fast in-memory lexical re-ranking to isolate the top $k=30$ candidate records per entity (reducing the comparison space by $> 99.999\%$).
2. **Stage 2 (Fine-Grained Classification & Threshold Tuning):** Computes a 17-dimensional similarity vector per candidate pair, executes inference using a multi-threaded LightGBM Gradient Boosted Decision Tree (GBDT), and maximizes the competition's **Macro $F_{0.5}$** metric via vectorized threshold tuning.

```mermaid
flowchart TD
    subgraph Data Layer
        S1[Source 1: Reference Records\ntrain_source1.tsv / test_source1.tsv]
        S2[Source 2 Records\n4.90M Records]
        S3[Source 3 Records\n5.28M Records]
    end

    subgraph Stage 1: Zero-RAM Candidate Blocking
        DB[(SQLite Disk Database\nindex.db\n10.18M Records | 63.2M Token Links)]
        S2 -->|Fast Bulk Insert| DB
        S3 -->|Fast Bulk Insert| DB
        S1 --> CleanTokens[Extract Informative Tokens\nFilter 352 Stop Tokens]
        CleanTokens --> BTreeQuery[Query Top 5 Rare Tokens\nLIMIT 400 per Token]
        BTreeQuery --> CandidatePool[Candidate Pool Union\n~500 - 1,500 Candidate IDs]
        CandidatePool --> LexicalRank[In-Memory Lexical Re-Ranker\n3x Name + 1x Addr + 2x First Word]
        LexicalRank --> TopK[Top 30 Candidate Pairs\nRecall: 60.0%]
    end

    subgraph Stage 2: Feature Engineering
        TopK --> PrecomputeS1[Precompute S1 Representation\nTokens, Digits, 3-Grams]
        PrecomputeS1 --> PairFeats[Extract 17 Similarity Features\nContainment, Jaccard, Exact Match, Interaction]
    end

    subgraph Stage 3: LightGBM ML Inference
        PairFeats --> LightGBM[LightGBM GBDT Classifier\nn_estimators=200, depth=6\nMulti-Threaded CPU Engine]
        LightGBM --> PredProbs[Raw Pairwise Probabilities]
        PredProbs --> ThresholdTuner[Vectorized Macro F0.5 Optimizer\n80 Grid Steps on Val Split]
        ThresholdTuner --> BinaryPreds[Filtered Entity Matches\nOptimal Threshold = 0.700]
    end

    subgraph Stage 4: Submission Outputs
        TopK --> OutputCand[candidate_pairs.tsv\nTop 30 Candidates per S1]
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
| **Source 1** | `S1-` | Deduplicated Reference | ~50,000 – 100,000 | Primary query entity set |
| **Source 2** | `S2-` | Candidate Source A | 4,900,000 | Multi-jurisdiction records |
| **Source 3** | `S3-` | Candidate Source B | 5,285,603 | Accounts for **51.7%** of all ground truth matches |

### Canonical Schema
All TSV files share standard fields:
- `entity_id`: Primary record identifier (e.g. `S1-844896591`, `S2-362218588`, `S3-11966366`).
- `business_name`: Text with legal abbreviations, typos, transliterations, and repeated words.
- `business_address`: Noisy street addresses, landmarks, missing zip codes.
- `country`: Open-set geographic identifier (`US`, `India`, `France`).

---

## 3. Stage 1: Zero-RAM Candidate Blocking & Retrieval

### 3.1 Persistent SQLite Database Architecture
To eliminate out-of-memory (OOM) failures while searching 10.18 million records, candidate data is maintained in an optimized SQLite database (`index.db`):
- **`records` table**: Stores `(record_id, name, address, country, dataset)` for all 10,185,603 candidate records across Source 2 and Source 3.
- **`token_index` table**: Stores `(token, entity_id)` holding 63,268,914 inverted index entries with a clustered B-Tree index on `token_index(token)`.

### 3.2 High-Throughput Token Filtering
1. **Diacritic & Accent Normalization**: Unicode NFKD decomposition strips diacritics (`é` $\rightarrow$ `e`).
2. **Noise & Domain Stripping**: Strips URLs (`www.*`, `.com`), punctuation, and legal entity suffixes (`inc`, `llc`, `pvt ltd`, `gmbh`, `sarl`).
3. **Generic Token Shield**: Pre-filters a curated set of **352 high-frequency tokens** (`GENERIC_TOKENS`) appearing in $> 25,000$ database records (e.g. `delhi`, `mumbai`, `services`, `center`, `limited`) to avoid costly multi-thousand disk page reads.

### 3.3 Rare-Token Pooling & Lexical Re-Ranking
Rather than taking an unranked, arbitrary cutoff from SQLite:
1. Query up to 5 non-generic search tokens from the S1 entity with `LIMIT 400` each.
2. Pool all retrieved candidate IDs into a set (typically 500 – 1,500 distinct IDs).
3. Score each candidate against the S1 record in memory using lightweight C-level string operations:
   $$\text{Score} = 3 \cdot |\text{Words}_{\text{name1}} \cap \text{Words}_{\text{name2}}| + |\text{Words}_{\text{addr1}} \cap \text{Words}_{\text{addr2}}| + 2 \cdot \mathbb{I}(\text{first\_word}_1 == \text{first\_word}_2)$$
4. Sort candidates by score descending and return the top $k=30$ candidate IDs.

> [!IMPORTANT]
> **Recall Impact**: Implementing lexical re-ranking alongside full Source 3 indexing increased the candidate pool recall from **21.48% to 60.0%**.

---

## 4. Stage 2: High-Speed Pairwise Feature Engineering

To evaluate the top 30 candidate pairs per entity at over 2,000 comparisons/second, the S1 reference attributes are precomputed once via `precompute_s1_features()`, avoiding redundant regex executions across candidates.

### 17-Dimensional Feature Vector
Each $(S1, \text{Candidate})$ pair generates 17 numerical features:

| Category | Feature Name | Description | Formula / Logic |
| :--- | :--- | :--- | :--- |
| **Name** | `name_jaccard` | Word token overlap | $\frac{\|T_{n1} \cap T_{n2}\|}{\|T_{n1} \cup T_{n2}\|}$ |
| **Name** | `name_char_jaccard` | 3-gram character overlap | Jaccard over character 3-grams |
| **Name** | `name_ratio` | 2-gram Dice string similarity | Fast character bigram overlap |
| **Name** | `name_sort_ratio` | Alphabetically sorted token similarity | Invariant to word transposition |
| **Name** | `name_exact` | Exact normalized equality | $\mathbb{I}(\text{name}_1 == \text{name}_2)$ |
| **Name** | `name_containment` | Abbreviation / prefix containment | $\frac{\|T_{n1} \cap T_{n2}\|}{\min(\|T_{n1}\|, \|T_{n2}\|)}$ |
| **Name** | `first_word_match` | Anchor brand token match | $\mathbb{I}(\text{first\_word}_1 == \text{first\_word}_2)$ |
| **Address** | `addr_jaccard` | Address token overlap | $\frac{\|T_{a1} \cap T_{a2}\|}{\|T_{a1} \cup T_{a2}\|}$ |
| **Address** | `addr_char_jaccard` | Address character 3-gram overlap | Jaccard over address 3-grams |
| **Address** | `addr_sort_ratio` | Sorted address token similarity | Invariant to address component order |
| **Address** | `addr_containment` | Address subset ratio | $\frac{\|T_{a1} \cap T_{a2}\|}{\min(\|T_{a1}\|, \|T_{a2}\|)}$ |
| **Interaction**| `both_match_score` | Joint agreement term | `name_jaccard` $\times$ `addr_jaccard` |
| **Hard Key** | `house_num_match` | Street/building number match | 1.0 (exact), 0.5 (overlap), 0.0 (none) |
| **Hard Key** | `phone_match` | Normalized digit match | Binary equality on numeric digits |
| **Hard Key** | `email_match` | Normalized email match | Case-insensitive email equality |
| **Geo** | `country_match` | Open-set geographic agreement | Binary equality on country string |
| **Origin** | `is_s2` | Source partition flag | 1.0 if Source 2, 0.0 if Source 3 |

---

## 5. Stage 3: LightGBM Classifier & Macro F0.5 Threshold Tuning

### 5.1 Leak-Proof Grouped Validation
To ensure validation results mirror leaderboard test evaluation:
- The validation split (80/20) is partitioned **strictly by Source 1 entity IDs**.
- All candidate pairs for a given S1 entity belong exclusively to either Train or Validation, preventing data leakage.

### 5.2 Multi-Threaded GBDT Training
- **Model:** `lightgbm.LGBMClassifier`
- **Parameters:** `n_estimators=200`, `learning_rate=0.05`, `max_depth=6`, `num_leaves=31`, `n_jobs=-1`.
- **Runtime:** Trains on 1.4 million pairwise rows in **~7 seconds** on CPU.
- **Fail-Safe Fallback:** If GPU acceleration is requested but no OpenCL runtime is detected, the engine catches `LightGBMError` and immediately routes computation to multi-core CPU without crashing.

### 5.3 Exact Competition Metric Optimization (Macro F0.5)
The competition evaluation metric is **Macro $F_{0.5}$**, which weights precision twice as heavily as recall:
$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

The model runs a vectorized threshold search across 80 probability bins ($0.10 \le \tau \le 0.90$):
1. For each threshold $\tau$, candidate predictions are grouped by S1 entity.
2. Precision, Recall, and $F_{0.5}$ are calculated per S1 entity group.
3. The threshold maximizing the arithmetic mean across all entities ($\tau^* = 0.700$) is selected.

---

## 6. Stage 4: Output Generation & Submission Validation

The pipeline writes two official TSV artifacts to `output/`:

### 6.1 `candidate_pairs.tsv`
Records the top candidates produced by the blocking stage before model pruning:
```tsv
source1_entity_id	candidate_entity_ids
S1-844896591	S2-362218588,S3-11966366,S2-692939662,...
S1-378978603	S2-124893121,S3-948532405,...
```

### 6.2 `matching_results.tsv` (Leaderboard Submission)
Contains final predicted entity matches:
```tsv
source1_entity_id	matched_entity_ids
S1-844896591	S2-362218588,S3-11966366
S1-378978603	S3-948532405
S1-133037285	
```
*(Singletons have an empty `matched_entity_ids` field).*

### 6.3 Compliance Verification
Outputs are validated using the official competition validator:
```bash
python3 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
Checks: 1 row per S1 entity, no duplicates, valid TSV delimiters, and candidate subset constraints.

---

## 7. Benchmark Evolution & Diagnostic Insights

| Stage | Candidate Pool Size | Candidate Recall | Mean Precision | Mean Recall | Macro $F_{0.5}$ Score | Key Innovations Added |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 4.90M (S2 only) | 21.48% | 46.90% | 24.11% | **0.3694** | Unranked candidate queries, missing Source 3, basic 11 features |
| **Optimized** | **10.18M (S2 + S3)** | **60.00%** | **81.25%** | **57.29% – 63.63%** | **0.7206 – 0.7418** | Full S3 indexing, lexical re-ranking, 17 features, Macro threshold tuning |

---

## 8. Step-by-Step Operational Runbook

### Environment Setup
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Verify dependencies
pip install -r code/requirements.txt
```

### Running Pipeline on Train Split (Validation Mode)
```bash
# Runs candidate extraction, feature computation, and model training
python3 code/src/main.py --sample-size 50000 --max-candidates 30 --split train
```
*Features are automatically cached to `output/train/cache_features_train_50000_30.pkl` for instant re-runs.*

### Running Pipeline on Test Split (Official Leaderboard Mode)
```bash
python3 code/src/main.py --split test --max-candidates 30
```

### Packaging Final Submission Archive
Per challenge guidelines, prepare the final zip submission package:
```bash
zip -r Antigravity_submission.zip \
    output/test/matching_results.tsv \
    output/test/candidate_pairs.tsv \
    code/ \
    Documentation_template.md
```
