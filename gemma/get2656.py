from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

dsn = "amuvarma/wikipedia-filtered-en-tokenised"

snapshot_download(
    repo_id=dsn,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

ds = load_dataset(dsn, split='train')

filtered_ds = ds.filter(
    lambda x: len(x["input_ids"]) > 2656,
    num_proc=60
)

print("Number of rows over 2656 tokens:", filtered_ds.num_rows)

