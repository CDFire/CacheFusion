import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Parameters
kv_cache_dir = "/data/xier2/bitplane/kv_cache/llama3_8b/wikitext/fp32"  # Directory with KV cache files
output_dir = "/data/xier2/bitplane/kv_cache/pairwise_results"  # Directory to save results
os.makedirs(output_dir, exist_ok=True)

beta = 0.5  # Weight for temporal similarity
alpha = 0.1  # Temporal decay factor

def load_kv_cache(layer_idx):
    """Load KV cache for a specific layer."""
    kv_file = os.path.join(kv_cache_dir, f"kv_cache_layer_{layer_idx}.pkl")
    if not os.path.exists(kv_file):
        raise FileNotFoundError(f"KV cache file for layer {layer_idx} not found: {kv_file}")
    
    with open(kv_file, "rb") as f:
        kv_cache = pickle.load(f)
    return kv_cache

def pad_to_match(arrays):
    """Pad arrays to match dimensions along the concatenation axis."""
    max_dim = max(a.shape[2] for a in arrays)  # Find the maximum size along the mismatched axis
    padded_arrays = []
    for array in arrays:
        pad_width = [(0, 0)] * array.ndim
        pad_width[2] = (0, max_dim - array.shape[2])  # Pad only the mismatched axis
        padded_arrays.append(np.pad(array, pad_width, mode='constant'))
    return np.concatenate(padded_arrays, axis=0)

def compute_temporal_similarity(t1, t2):
    """Compute temporal similarity based on token indices."""
    return np.exp(-alpha * abs(t1 - t2))

def calculate_pairwise_similarities(layer_idx, kv_cache, beta):
    """Calculate pairwise similarities for all tokens."""
    # Pad keys and values to ensure dimensions match
    keys = pad_to_match(kv_cache["keys"])
    num_tokens = keys.shape[0]

    results = []  # To store pairwise results

    # Compute semantic similarity (cosine similarity)
    semantic_sim = cosine_similarity(keys.reshape(len(keys), -1))

    # Calculate pairwise temporal and hybrid similarities
    for i in range(num_tokens):
        for j in range(num_tokens):
            if i == j:  # Skip self-comparisons
                continue
            temporal_sim = compute_temporal_similarity(i, j)
            hybrid_sim = beta * temporal_sim + (1 - beta) * semantic_sim[i, j]

            results.append({
                "layer": layer_idx,
                "token1_idx": i,
                "token2_idx": j,
                "temporal_similarity": temporal_sim,
                "semantic_similarity": semantic_sim[i, j],
                "hybrid_similarity": hybrid_sim,
            })

    return results

# Main processing
layer_indices = range(10)  # Adjust based on your model
for layer_idx in layer_indices:
    print(f"Processing layer {layer_idx}...")
    kv_cache = load_kv_cache(layer_idx)
    pairwise_results = calculate_pairwise_similarities(layer_idx, kv_cache, beta)

    # Save pairwise results to CSV
    output_file = os.path.join(output_dir, f"layer_{layer_idx}_pairwise.csv")
    df = pd.DataFrame(pairwise_results)
    df.to_csv(output_file, index=False)
    print(f"Pairwise similarity results for layer {layer_idx} saved to {output_file}")

print("Pairwise similarity calculations completed.")
