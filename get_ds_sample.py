dsn = "amuvarma/emilia-snac-merged-all-TTS-grouped-8192"
from datasets import load_dataset
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=dsn,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

ds = load_dataset(dsn, split='train')
ds = ds.shuffle(seed=42)

sample_size = 1000

sampled_ds = ds.select(range(sample_size))
sampled_ds.push_to_hub(f"{dsn}-sample-{sample_size}")