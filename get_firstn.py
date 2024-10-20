from datasets import load_dataset
dsn = "amuvarma/2.3m-llama-wdups"
ds = load_dataset(dsn, split='train')
number_of_rows_to_reupload = 50000
ds = ds.select(range(number_of_rows_to_reupload))
ds.push_to_hub('amuvarma/50k-llama-wdups')