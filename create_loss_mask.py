from datasets import load_dataset
dsn = "amuvarma/emilia-snac-merged-all-TTS-grouped-8192-sample-1000"
ds = load_dataset(dsn, split='train')
def mask_out_ids(example):

    threshold = 128266 + 4096      # 132362
    lower_bound = 128266 + 4 * 4096  # 144650
    upper_bound = 128266 + 5 * 4096  # 148746
    
    labels = []
    for token_id in example['input_ids']:
        # if token_id < threshold or (lower_bound <= token_id < upper_bound):
        if token_id < threshold or (lower_bound <= token_id < upper_bound):
            labels.append(token_id)
        else:
            labels.append(-100)
    
    example['labels'] = labels
    
    # Create an 'attention_mask' with 1s for all tokens
    example['attention_mask'] = [1] * len(example['input_ids'])
    
    
    return example


masked_ds = ds.map(mask_out_ids, num_proc=64)

masked_ds.push_to_hub(f"{dsn}-masked-row-1")