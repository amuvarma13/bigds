from datasets import load_dataset
from huggingface_hub import snapshot_download
dsn = "amuvarma/emilia-snac-merged-18m"
snapshot_download(
        repo_id=dsn,
        repo_type="dataset",   
        revision="main",        
        max_workers=64,     
)

ds = load_dataset(dsn, split='train')

def map_function(example):
    # Get the codes_list from the example
    codes = example['codes_list']
    
    # Keep only every 7th element starting from index 0
    filtered_codes = codes[::7]
    
    # Create a new example with the filtered codes
    example['codes_list'] = filtered_codes
    
    return example

# Apply the map function to your dataset
filtered_dataset = ds.map(map_function)
filtered_dataset.push_to_hub("amuvarma/emilia-snac-merged-18m-mod7")