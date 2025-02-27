from datasets import load_dataset

from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amphion/Emilia-Dataset"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",   
    revision="main",        
    max_workers=64, 
    allow_patterns=["Emilia/EN/*.tar"],       
)

load_dataset(repo_id, split="train")
