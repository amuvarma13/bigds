from datasets import load_dataset
from huggingface_hub import snapshot_download
dsn = "amuvarma/snac-raw-10m"
def _load_dataset(dataset_name):
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",   
        revision="main",        
        max_workers=64         
    )
    return load_dataset(dataset_name, split="train")

dataset = _load_dataset(dsn)
print(dataset)
