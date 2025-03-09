from datasets import load_dataset
dsn = "amuvarma/emilia-snac-merged-all-TTS-grouped-8192-sample-1000"
ds = load_dataset(dsn, split='train')

def mask_out_ids(example):
    # Create a 'labels' column that contains input_ids values only if they're less than 132362
    example['labels'] = [token_id if token_id < 132362 else -100 for token_id in example['input_ids']]
    
    # Create an 'attention_mask' with 1s for all tokens
    example['attention_mask'] = [1] * len(example['input_ids'])

    
    return example


masked_ds = ds.map(mask_out_ids, num_proc=64)

masked_ds.push_to_hub(f"{dsn}-masked-row-1")