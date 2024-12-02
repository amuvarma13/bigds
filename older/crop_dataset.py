from datasets import load_dataset

dsn = "amuvarma/6-layer-crossmodal-750k-2"
ds = load_dataset(dsn)

# first shuffling the dataset
ds = ds.shuffle(seed=42)

# get the first 500k rows 

ds_500k = ds['train'].select(range(500000))

# push to hub
ds_500k.push_to_hub("amuvarma/6-layer-xmodal-500k-0")