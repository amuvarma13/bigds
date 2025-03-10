dsn = "amuvarma/emilia-snac-merged-18m"
from datasets import load_dataset
from huggingface_hub import snapshot_download

push_name = f"{dsn}-smol"

snapshot_download(
    repo_id=dsn,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

ds = load_dataset(dsn, split='train')

constant_to_subtract = 128256-49152

def subtract_constant(example):
    example["codes_list"] = [x - constant_to_subtract for x in example["codes_list"]]
    return example

ds = ds.map(subtract_constant, num_proc=64)

ds.push_to_hub(push_name)