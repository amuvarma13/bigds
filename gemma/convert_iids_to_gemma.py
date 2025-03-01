from datasets import load_dataset
from huggingface_hub import snapshot_download
dsn = "amuvarma/snac-raw-10m"
def _load_dataset(dataset_name):
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",   
        revision="main",        
        max_workers=80         
    )
    return load_dataset(dataset_name, split="train")

dataset = _load_dataset(dsn)
print(dataset)

amount_to_add = 127744
import numpy as np

def add_offset(batch):
    # Convert to a NumPy array (only works efficiently if the inner lists have the same length)
    codes_array = np.array(batch["codes"])
    # Vectorized addition
    codes_array += 127744
    # Convert back to a list of lists
    batch["codes"] = codes_array.tolist()
    return batch

# Apply the function with batched=True for efficiency
dataset = dataset.map(add_offset, batched=True)
dataset.push_to_hub(f"{dsn}-gemma")
