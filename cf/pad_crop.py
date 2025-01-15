from datasets import load_dataset
import os


dsn = "amuvarma/snac-2m-tts-2g"
push_name = "amuvarma/snac-2m-tts-2g-3328l"


ds = load_dataset(dsn, split='train')

max_length = 3328
pad_token = 128263

num_cpus = os.cpu_count()

num_processes = max(1, int(num_cpus * 0.75))

def pad_and_create_mask(example):
    if len(example['input_ids']) > max_length:
        example['input_ids'] = example['input_ids'][:max_length]
        example['attention_mask'] = [1] * max_length
        example["labels"] = example["input_ids"][:max_length]
    else:
        padding_length = max_length - len(example['input_ids'])
        example['attention_mask'] = [1] * len(example['input_ids']) + [0] * padding_length
        example["labels"] = example["input_ids"] + [-100] * padding_length
        example['input_ids'] = example['input_ids'] + [pad_token] * padding_length
        

    return example
ds_1 = ds.map(pad_and_create_mask, num_proc=num_processes)

ds_1.push_to_hub(push_name)