from datasets import load_dataset
import os


dsn = "amuvarma/text-messages-6m-processed-1-2g"
push_name = "amuvarma/text-messages-6m-processed-1-2g-8192l"


ds = load_dataset(dsn, split='train')

max_length = 8192

num_cpus = os.cpu_count()

num_processes = max(1, int(num_cpus * 0.75))

def pad_and_create_mask(example):
    if len(example['input_ids']) > max_length:
        example['input_ids'] = example['input_ids'][:max_length]
        example['attention_mask'] = [1] * max_length
        example["labels"] = example["input_ids"][:max_length]
    else:
        example['attention_mask'] = [1] * len(example['input_ids'])
        example["labels"] = example["input_ids"]
        example['input_ids'] = example['input_ids']

    return example


ds_1 = ds.map(pad_and_create_mask, num_proc=num_processes)

ds_1.push_to_hub(push_name)