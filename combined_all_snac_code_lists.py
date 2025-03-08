dss = ["amuvarma/emilia-snac-merged-amuvarma", 
       "amuvarma/emilia-snac-merged-CanopyLabs", 
       "amuvarma/emilia-snac-merged-eliasfiz", 
       "amuvarma/emilia-snac-merged-akv13", 
       "amuvarma/emilia-snac-merged-CanopyElias", 
       "amuvarma/emilia-snac-merged-CanopyLabsElias"

       ]

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download
all_ds = []
for dsn in dss:
    snapshot_download(
        repo_id=dsn,
        repo_type="dataset",   
        revision="main",        
        max_workers=64,     
    )

    ds = load_dataset(dsn, split='train')
    all_ds.append(ds)
    print(f"Loaded {dsn}")

ds = concatenate_datasets(all_ds)

print(ds)
ds.push_to_hub("amuvarma/emilia-snac-merged-18m")


