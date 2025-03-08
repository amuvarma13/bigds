from datasets import Dataset, load_dataset, concatenate_datasets
from itertools import chain
import os
import time

account_name = "amuvarma"
dsn = f"amuvarma/emilia-snac-merged-18m-mod7-delay-TTS"

dataset = load_dataset(dsn, split='train')
print(dataset)
dataset_length = 17114684
partition_size = 503000
chunk_size = 8192


dataset_length = len(dataset)

dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['input_ids']])


num_partitions = dataset_length // partition_size
print(f"Number of partitions: {num_partitions}")
processed_partitions = []

for i in range(num_partitions):
    print(f"Partition {i}")
    start = i * partition_size
    end = (i + 1) * partition_size 
    partition = dataset.select(range(start, end))
    start_time = time.time()


    all_tokens = list(chain.from_iterable(partition["input_ids"]))


    num_chunks = len(all_tokens) // chunk_size  # This drops any leftover tokens
    chunks = [all_tokens[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]


    new_dataset = Dataset.from_dict({"input_ids": chunks})
    end_time = time.time()
    print(f"Time taken to chunk: {end_time - start_time}")
    # push_name = f"{dsn}-{i}-of-{dataset_length//partition_size}"
    # new_dataset.push_to_hub(push_name)
    processed_partitions.append(new_dataset)


combined_dataset = concatenate_datasets(processed_partitions)
combined_dataset.push_to_hub(f"amuvarma/emilia-snac-merged-18m-mod7-delay-TTS-all-rows-8192")