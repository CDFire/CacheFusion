## CacheFusion 


### Scripts Overview

#### 1. `capture_kvcache.py`
- **Purpose**: Captures KV cache data, computes pairwise similarities, and clusters tokens based on hybrid similarity.
- **Key Features**:
  - Computes temporal, semantic, and hybrid similarities.
  - Clusters tokens into pages of up to `page_size` tokens.
  - Outputs detailed cluster information to CSV files.
- **Output**: `layer_{layer_idx}_cluster_with_similarities.csv` containing:
  - Cluster ID, token pairs, temporal/semantic/hybrid similarities.

---

#### 2. `cluster.py`
- **Purpose**: Performs clustering of tokens in KV caches for experiments.
- **Key Features**:
  - Similar to `capture_kvcache.py`, focuses on clustering logic.
  - Computes similarities and clusters tokens into pages.
- **Output**: `layer_{layer_idx}_cluster_with_similarities.csv`.

---

#### 3. `collect_pairwise_kv_token_hybrid_similarity.py`
- **Purpose**: Computes pairwise temporal, semantic, and hybrid similarities for all tokens without clustering.
- **Key Features**:
  - Handles large KV cache files with dimensional consistency.
  - Saves pairwise similarities for analysis.
- **Output**: `layer_{layer_idx}_pairwise.csv` containing:
  - Token pair indices, temporal/semantic/hybrid similarities.

