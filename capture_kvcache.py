import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pickle
import os

model_path = "/data/xier2/mixture_of_depths/meta-llama-3-8B" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

model.config.use_cache = True

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

output_dir = "/data/xier2/bitplane/kv_cache/llama3_8b/wikitext/fp32"
os.makedirs(output_dir, exist_ok=True)

layer_kv_cache = {}

precision = torch.float32 

num_examples = 10000
for i, example in enumerate(dataset):
    if i >= num_examples:
        break

    input_text = example['text']
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        kv_cache = outputs.past_key_values

    for layer_idx, (keys, values) in enumerate(kv_cache):
        if layer_idx not in layer_kv_cache:
            layer_kv_cache[layer_idx] = {"keys": [], "values": []}
        
        keys_converted = keys.to(precision).to(torch.float32).detach().cpu().numpy()
        values_converted = values.to(precision).to(torch.float32).detach().cpu().numpy()
        
        layer_kv_cache[layer_idx]["keys"].append(keys_converted)
        layer_kv_cache[layer_idx]["values"].append(values_converted)

    print(f"Processed {i + 1} entries")

for layer_idx, kv in layer_kv_cache.items():
    layer_file = os.path.join(output_dir, f"kv_cache_layer_{layer_idx}.pkl")
    
    with open(layer_file, "wb") as f:
        pickle.dump(kv, f)

print("All KV caches stored successfully.")
