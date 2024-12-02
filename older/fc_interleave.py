from datasets import load_dataset
ds = load_dataset("amuvarma/1m-fac_0")

def process_facodec(example):
    facodec_1 = example['facodec_1']
    facodec_0 = example['facodec_0']
    
    # Interleave and add required values
    interleaved = []
    for f1, f0 in zip(facodec_1, facodec_0):
        interleaved.append(f1 + 256003)
        interleaved.append(f0 + 258000)
    
    example['interleaved_facodec'] = interleaved
    return example

def pad_and_mask(batch):
    # Find the maximum length in the batch
    max_length = max(len(seq) for seq in batch['interleaved_facodec'])
    
    # Pad sequences and create attention masks
    input_ids = []
    attention_masks = []
    
    for seq in batch['interleaved_facodec']:
        padded_seq = seq + [0] * (max_length - len(seq))
        mask = [1] * len(seq) + [0] * (max_length - len(seq))
        
        input_ids.append(padded_seq)
        attention_masks.append(mask)
    
    batch['input_ids'] = input_ids
    batch['attention_mask'] = attention_masks
    
    return batch

# Apply the interleaving function to the dataset
ds['train'] = ds['train'].map(process_facodec)

# Apply padding and create attention masks
ds['train'] = ds['train'].map(pad_and_mask, batched=True, batch_size=1000)


ds.push_to_hub("fc_0_1-interleaved-1m")