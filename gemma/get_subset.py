from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

dsn = "amuvarma/text-messages-6m"

snapshot_download(
    repo_id=dsn,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

ds = load_dataset(dsn, split='train')
