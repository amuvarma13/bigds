from datasets import load_dataset
from huggingface_hub import snapshot_download
dsn = "amuvarma/luna-48k-b7CS6GHVkhPt9lmufYchXdy7eLo1-enhanced-clipped-snacced"
snapshot_download(
        repo_id=dsn,
        repo_type="dataset",   
        revision="main",        
        max_workers=64,     
)

ds = load_dataset(dsn, split='train')

#filter out all rows without codes list  
ds = ds.filter(lambda x: x["codes_list"] is not None, num_proc=64)
ds = ds.filter(lambda x: len(x["codes_list"]) > 0, num_proc=64)

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
filtered_dataset.push_to_hub("amuvarma/luna-enh-clip-snac-mod7-1-5")