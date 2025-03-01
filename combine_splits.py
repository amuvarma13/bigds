from huggingface_hub import snapshot_download
from datasets import load_dataset, concatenate_datasets
repo_name = "CanopyLabsElias"
repo_id = f"{repo_name}/emilia-snac"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

ds = load_dataset(repo_id)

all_split_datasets = [ds[split_name] for split_name in ds.keys()]

# Merge them into a single Dataset
merged_dataset = concatenate_datasets(all_split_datasets)

print(merged_dataset)

merged_dataset.push_to_hub(f"amuvarma/emilia-snac-merged-{repo_name}")

 