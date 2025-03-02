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

accs = ["amuvarma", "CanopyLabs", "CanopyLabsElias", "eliasfiz", "CanopyElias", "akv13"]

dsns = [f"amuvarma/emilia-snac-merged-{acc}-gemma-TTS-grouped-2656" for acc in accs]

datasets = [_load_dataset(dsn) for dsn in dsns]

combined_dataset = concatenate_datasets(datasets)
combined_dataset = combined_dataset.shuffle(seed=42)

combined_dataset.push_to_hub(f"amuvarma/emilia-snac-merged-all-gemma-TTS-grouped-2656")
