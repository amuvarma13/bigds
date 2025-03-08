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
    codes = example['codes_list']
    filtered_codes = []
    
    for i in range(0, len(codes), 7):
        # Append the 0th element of the block
        filtered_codes.append(codes[i])
        
        # Append the 4th element if it exists
        if i + 4 < len(codes):
            filtered_codes.append(codes[i + 4])
    
    example['codes_list'] = filtered_codes
    return example


# Apply the map function to your dataset
filtered_dataset = ds.map(map_function, num_proc=64)
filtered_dataset.push_to_hub("amuvarma/emilia-snac-merged-18m-mod7")