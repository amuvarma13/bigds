from datasets import load_dataset
dsn = "amuvarma/snac-2m-tts-2g"
push_name = "amuvarma/snac-2m-tts-2g-3328l"
import os

ds = load_dataset(dsn, split='train')

max_length = 3328
pad_token = 128263

def pad_and_create_mask(example):
    if len(example['input_ids']) > max_length:
        example['input_ids'] = example['input_ids'][:max_length]
        example['attention_mask'] = [1] * max_length
    else:
        padding_length = max_length - len(example['input_ids'])
        example['attention_mask'] = [1] * len(example['input_ids']) + [0] * padding_length
        example['input_ids'] = example['input_ids'] + [pad_token] * padding_length

    return example
ds_1 = ds.map(pad_and_create_mask)

def preprocess_function(examples):
    examples['labels'] = [
        [(token_id if token_id != pad_token else -100) for token_id in input_ids]
        for input_ids in examples['input_ids']
    ]
    return examples

num_cpus = os.cpu_count()

num_processes = max(1, int(num_cpus * 0.75))

ds_2 = ds_1.map(
    preprocess_function,
    batched=True,
    num_proc=num_processes,
    desc="Preprocessing dataset"
)

# Save the dataset

ds_2.push_to_hub(push_name)