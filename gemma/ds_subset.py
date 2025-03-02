from datasets import load_dataset
import os
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

user_name = "CanopyElias"
dsn = f"amuvarma/emilia-snac-merged-{user_name}-gemma-TTS"

snapshot_download(
    repo_id=dsn,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

ds = load_dataset(dsn, split='train')

print(ds)

ds = ds.select(range(30000))

ds.push_to_hub(f"{dsn}-30k")