from datasets import load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download

def _load_dataset(dsn):
    snapshot_download(
        repo_id=dsn,
        repo_type="dataset",   
        revision="main",        
        max_workers=64,     
    )
    return load_dataset(dsn, split='train')

accs = ["DyqtR9tWy9TfTeMyq59s6Ke4jEq1", "6QYh18mJTKevBkRLRZqsfiPc7oH2", "U3czDRQZoTRFBAK3z6FVc0V5EYC3", "b7CS6GHVkhPt9lmufYchXdy7eLo1"]

dsns = [f"amuvarma/luna-48k-{acc}-enhanced" for acc in accs]

datasets = [_load_dataset(dsn) for dsn in dsns]

combined_dataset = concatenate_datasets(datasets)
combined_dataset = combined_dataset.shuffle(seed=42)

combined_dataset.push_to_hub(f"amuvarma/luna-48k-full-enhanced")
