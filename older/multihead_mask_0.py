from datasets import load_dataset

ds_name = "eliasfiz/audio_content_prosody_part1"
ds = load_dataset(ds_name)

def create_embed_head_mask(example):
    embed_head_mask = [0] * len(example['input_ids'])
    for i, id in enumerate(example['input_ids']):
        if 256010 <= id <= 257034:
            embed_head_mask[i] = 1
    example['embed_head_mask'] = embed_head_mask
    return example

# Assuming your dataset is called 'ds'
ds = ds.map(create_embed_head_mask)

def create_text_labels(example):
    input_ids = example['input_ids']
    text_labels = [-100] * len(input_ids)

    for i in range(len(input_ids) - 1):
        if not (256010 <= input_ids[i] <= 257034):
            text_labels[i] = input_ids[i] 
    
    example['text_labels'] = text_labels
    return example

# Assuming your dataset is called 'ds'
ds = ds.map(create_text_labels)

def process_dataset(example):
    input_ids = example['input_ids']
    
    for i in range(len(input_ids)):
        if 256010 <= input_ids[i] <= 257034:
            input_ids[i] -= 256010  # Subtract 256010 from the input_id
    
    example['input_ids'] = input_ids  # Update input_ids in the example
    return example

# Assuming your dataset is called 'ds'
ds = ds.map(process_dataset)

def create_audio_labels(example):
    input_ids = example['input_ids']
    text_labels = example['text_labels']
    audio_labels = [-100] * len(input_ids)
    
    for i in range(len(input_ids)):
        if text_labels[i] == -100:
            audio_labels[i] = input_ids[i]
    
    example['audio_labels'] = audio_labels
    return example

# Assuming your dataset is called 'ds'
ds = ds.map(create_audio_labels)

def replace_specific_zeros_with_negative_hundred(example):
    text_labels = example['text_labels']
    audio_labels = example['audio_labels']

    def process_labels(labels):
        new_labels = labels.copy()
        for i in range(1, len(labels) - 1):
            if labels[i] == 0 and (labels[i-1] == 0 or labels[i+1] == 0):
                new_labels[i] = -100
        return new_labels

    example['text_labels'] = process_labels(text_labels)
    example['audio_labels'] = process_labels(audio_labels)
    return example

# Assuming your dataset is called 'ds'
ds = ds.map(replace_specific_zeros_with_negative_hundred)

ds.push_to_hub("amuvarma/text-content-470k-0")