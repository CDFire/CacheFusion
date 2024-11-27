## CacheFusion 


### Scripts Overview

#### 1. `capture_kvcache.py`
- **Purpose**: Captures the KV cache from a model and stores it to disk for further analysis.
- **Key Features**:
  - Extracts KV caches for each transformer layer.
  - Saves the KV data as `.pkl` files for efficient storage.
- **Output**: Files named `kv_cache_layer_{layer_idx}.pkl` in the specified directory.

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

