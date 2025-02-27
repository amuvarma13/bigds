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
dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en", streaming=True)
from datasets import Dataset

# Suppose `iterable_dataset` yields dictionaries for each example.
data = list(dataset)  # Convert iterator to list of examples
hf_dataset = Dataset.from_list(data)
print(hf_dataset)
