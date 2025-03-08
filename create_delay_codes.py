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
    # Make a copy so we can modify positions without losing the original values.
    new_codes = codes.copy()
    
    # For every index that's a multiple of 7, starting at 0:
    for i in range(0, len(codes), 7):
        # For index 0, there’s no valid “-7” target so we replace it with the code 7 positions later (if available)
        if i == 0:
            if i + 7 < len(codes):
                new_codes[i] = codes[i + 7]
            else:
                # If there isn’t an element at i+7, mark it for removal.
                new_codes[i] = None
        else:
            # For any other index i that is a multiple of 7, shift the code back 7 positions.
            # That is, place the code originally at position i into position i - 7.
            new_codes[i - 7] = codes[i]
    
    # Now remove any None entries (which might appear if a block didn’t have enough elements to shift in)
    filtered_codes = [code for code in new_codes if code is not None]
    
    example['codes_list'] = filtered_codes
    return example




# Apply the map function to your dataset
filtered_dataset = ds.map(map_function, num_proc=64)
filtered_dataset.push_to_hub("amuvarma/emilia-snac-merged-18m-mod7-delay")