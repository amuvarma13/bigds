from huggingface_hub import snapshot_download
from datasets import load_dataset, concatenate_datasets

all_repo_names = ["eliasfiz/podcast-snac"]
all_datasets = []
def merge_split(repo_name):
    repo_id = f"{repo_name}"

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",   
        revision="main",        
        max_workers=64,     
    )

    ds = load_dataset(repo_id)

    all_split_datasets = [ds[split_name] for split_name in ds.keys()]

    merged_dataset = concatenate_datasets(all_split_datasets)

    all_datasets.append(merged_dataset)


merge_split(all_repo_names[0])

# Merge all datasets into a single Dataset
merged_dataset = concatenate_datasets(all_datasets)

print(merged_dataset)

merged_dataset.push_to_hub(f"amuvarma/podcast-all")

 