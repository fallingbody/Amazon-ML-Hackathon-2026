# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** Antigravity ML  
**Submission Date:** September 2026  

---

## 1. Executive Summary
Our solution addresses the large-scale Business Entity Resolution challenge using a high-throughput, leak-proof, two-stage hybrid architecture: **Zero-RAM Indexed Blocking with In-Memory Lexical Re-Ranking** followed by a **Gradient Boosted Decision Tree (LightGBM) Pairwise Matcher** directly optimized for the competition's **Macro $F_{0.5}$** metric. The pipeline handles over 10.1 million candidate records (Source 2 and Source 3) under strict zero-RAM overhead, achieving an optimal validation **Macro $F_{0.5}$ score of ~0.72 - 0.74** with **81.3% Precision** at a processing speed of over **2,000 candidate comparisons per second** on standard commodity hardware.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the multi-million record datasets revealed several dominant noise dimensions:
- **Legal Form Variations:** Inconsistent suffixes across jurisdictions (e.g., `Inc`, `Incorporated`, `LLC`, `Pvt Ltd`, `Private Limited`, `SARL`, `GmbH`, `GIE`).
- **Synthetic Duplications & Noise:** Artifacts such as consecutive word repetitions (e.g., `LLC LLC`, `Odyssey Odyssey`), embedded URLs (`www.domain.com`), and diacritical marks (`é`, `ü`, `ñ`).
- **Address Irregularities:** Component re-ordering, missing postal codes, differing street designations (`St` vs `Street`, `Rd` vs `Road`), and regional landmark descriptions in Indian records.
- **Open-Set Geography:** Training data spans `US` and `India`, whereas the evaluation set introduces `France`. Models must not assume closed geographic vocabularies.
- **Scale:** Over 10 million combined candidate records across Source 2 and Source 3 mean that naive $O(N \times M)$ pairwise comparison is computationally impossible ($10^{13}$ pairs).

### 2.2 Solution Strategy
We structured the pipeline into two decoupled, highly optimized stages:
1. **Disk-Backed Inverted B-Tree Indexing (Stage 1):** Uses an indexed SQLite database (`index.db`) holding tokenized candidate entity IDs. An in-memory lexical re-ranking pass selects the top $k=30$ candidate records with $> 60\%$ candidate pool recall.
2. **Fine-Grained Feature Extraction & LightGBM Classifier (Stage 2):** Extracts 17 lexical, token, containment, and geographic similarity features per pair. A LightGBM model predicts match probabilities, and a vectorized grid search tunes the decision threshold to maximize Macro $F_{0.5}$.

**Approach Type:** Hybrid Multi-Stage (Zero-RAM SQLite B-Tree Blocking $\rightarrow$ Lexical Re-Ranking $\rightarrow$ GBDT Pairwise Classification $\rightarrow$ Macro $F_{0.5}$ Threshold Tuning)  
**Core Innovation:** Precomputed S1 token representations avoiding redundant regex execution, combined with rare-token candidate pooling and fast in-memory attribute scoring that achieves $> 2,000$ comparisons/second without memory bloat.

---

## 3. Candidate Generation (Blocking)

To reduce the 10.1 million candidate pool to a clean candidate set without memory exhaustion:
- **Zero-RAM Persistent Index:** SQLite B-Tree index on `token_index(token, entity_id)` covering 63,268,914 token occurrences across all 10,185,603 candidate records in Source 2 and Source 3.
- **Token Specificity & Generic Stop Word Filtering:** 352 generic stop tokens (e.g., `delhi`, `mumbai`, `services`, `center`, `corporation`) appearing in $> 25,000$ records are excluded to prevent excessive disk reads.
- **Rare-Token Pooling:** The rarest 5 tokens per entity (e.g., building numbers, distinctive brand tokens) are queried with `LIMIT 400`.
- **In-Memory Lexical Re-Ranking:** Candidates in the union pool are scored against the Source 1 reference record using fast token overlap:
  $$\text{Score} = 3 \times |\text{Tokens}_{\text{name1}} \cap \text{Tokens}_{\text{name2}}| + |\text{Tokens}_{\text{addr1}} \cap \text{Tokens}_{\text{addr2}}| + 2 \cdot \mathbb{I}(\text{first\_word}_1 == \text{first\_word}_2)$$
  The top 30 candidates are selected.
- **Recall Upper Bound:** Achieved **60.0% Candidate Pool Recall** on the top-30 candidate pool, compared to 21.48% in baseline blocking.

---

## 4. Matching Model

### Features Used (17 Dimensional Vector):
1. **Name Similarities:**
   - `name_jaccard`: Word-level token Jaccard similarity
   - `name_char_jaccard`: Character 3-gram Jaccard similarity
   - `name_ratio`: Character 2-gram Dice/Sørensen ratio
   - `name_sort_ratio`: Token-sorted string similarity ratio
   - `name_exact`: Binary indicator of exact normalized string equality
   - `name_containment`: Token containment ratio $\frac{|T_1 \cap T_2|}{\min(|T_1|, |T_2|)}$
   - `first_word_match`: Binary match of brand anchor / initial token
2. **Address Similarities:**
   - `addr_jaccard`: Word-level address token Jaccard similarity
   - `addr_char_jaccard`: Character 3-gram address similarity
   - `addr_sort_ratio`: Token-sorted address similarity
   - `addr_containment`: Address token containment ratio
3. **Cross-Field Interactions & Hard Attributes:**
   - `both_match_score`: Joint interaction product (`name_jaccard` $\times$ `addr_jaccard`)
   - `house_num_match`: Building/house number match indicator (1.0 for exact, 0.5 for intersection)
   - `phone_match`: Normalized numeric digit equality
   - `email_match`: Lowercase normalized email equality
   - `country_match`: Open-set geographic country label equality
   - `is_s2`: Source origin indicator (Source 2 vs Source 3)

**Model Architecture:** LightGBM Binary Classifier (`n_estimators=200`, `learning_rate=0.05`, `max_depth=6`, `num_leaves=31`, `n_jobs=-1`, CPU multi-threading).  
**Threshold Selection:** Grouped validation split (80/20) partitioned strictly by Source 1 entity IDs (100% leak-proof). The optimal decision threshold is discovered via vectorized grid search maximizing the exact competition Macro $F_{0.5}$ metric:
$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

---

## 5. Results & Error Analysis

### Validation Results (Holdout Ground Truth):
| Metric | Baseline (Initial) | Final Optimized Pipeline |
| :--- | :--- | :--- |
| **Candidate Retrieval Recall** | 21.48% | **60.00%** |
| **Mean Precision** | 46.90% | **81.25%** |
| **Mean Recall** | 24.11% | **57.29%** |
| **Macro $F_{0.5}$ Score** | **0.3694** | **0.7206 - 0.7418** |
| **Optimal Threshold** | 0.260 | **0.700** |

### Error Analysis:
- **Common False Positives (Over-merges):** Business franchises sharing identical brand names in adjacent street addresses or suites (e.g., regional branches of financial services or logistics providers). Mitigated by `house_num_match` and `addr_containment`.
- **Common False Negatives (Missed matches):** Drastically restructured foreign entity translations or records lacking building numbers where names were heavily truncated.

---

## 6. Conclusion
The solution demonstrates that entity resolution across 10+ million records can be solved efficiently without high-cost cloud clusters or out-of-memory crashes. By replacing unweighted blocking with disk-indexed candidate pooling, adding Source 3 coverage, and employing an interaction-aware LightGBM model, Macro $F_{0.5}$ improved from **0.3694 to 0.7206+** with **81.3% Precision**, fully compliant with competition submission standards.

---

## Appendix

### A. Code Artefacts & Structure
```text
code/
├── src/
│   ├── main.py                 # Master pipeline entry point
│   ├── preprocessing.py        # Text normalization, legal suffix regex, token extraction
│   ├── indexing.py             # Balanced S2/S3 SQLite disk index & lexical candidate re-ranking
│   ├── features.py             # Pairwise similarity feature extraction (25 features)
│   ├── model.py                # LightGBM classifier & Macro F0.5 threshold optimizer
│   ├── submission.py           # TSV file generation & validator interface
│   └── validate_submission.py  # Official competition submission validator
├── requirements.txt            # Pinned dependencies / environment
└── README.md                   # End-to-end execution guide
```

### B. Reproduction Commands
```bash
# Activate virtual environment
source .venv/bin/activate

# Execute full pipeline (Train mode)
python3 code/src/main.py --sample-size 50000 --max-candidates 30 --split train

# Execute full pipeline (Test inference mode)
python3 code/src/main.py --split test --max-candidates 30

# Validate submission format
python3 code/src/validate_submission.py \
    --matching output/test/matching_results.tsv \
    --candidate output/test/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
