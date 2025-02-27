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

path = "Emilia/EN/*.tar"
dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")
print(dataset)

subds = dataset.select(range(100))

subds.push_to_hub("amuvarma/Emilia-Dataset-EN-200")
