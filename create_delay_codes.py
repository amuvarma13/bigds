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
# ds = ds.select(range(1000))
def map_function(example):
    codes = example['codes_list']
    filtered_codes = []
    
    # Start at index 7 to “shift back” every 7th element (i.e. skip the original 0th)
    for i in range(7, len(codes), 7):
        # Append the element that was at the start of the block (now shifted back by 7 positions)
        filtered_codes.append(codes[i])
    
    example['codes_list'] = filtered_codes
    return example




# Apply the map function to your dataset
filtered_dataset = ds.map(map_function, num_proc=64)
filtered_dataset.push_to_hub("amuvarma/emilia-snac-merged-18m-mod7-delay")