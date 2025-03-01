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
def add_offset(batch):
    # batch["codes"] is a list of lists; add 127744 to each integer
    batch["codes"] = [[code + amount_to_add for code in codes_list] for codes_list in batch["codes"]]
    return batch

# Apply the function with batched=True for efficiency
dataset = dataset.map(add_offset, batched=True)

dataset.push_to_hub(f"{dsn}-gemma")
