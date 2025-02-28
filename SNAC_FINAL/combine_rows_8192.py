from datasets import Dataset, load_dataset
from itertools import chain
import os
import time


dsn = "amuvarma/emilia-30k-TTS"
dataset = load_dataset(dsn, split='train')
print(dataset)

dataset = dataset.select(range(10000))

start_time = time.time()
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['input_ids']])


all_tokens = list(chain.from_iterable(dataset["input_ids"]))

chunk_size = 4096
num_chunks = len(all_tokens) // chunk_size  # This drops any leftover tokens
chunks = [all_tokens[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]


new_dataset = Dataset.from_dict({"input_ids": chunks})
end_time = time.time()
print(f"Time taken to chunk: {end_time - start_time}")
push_name = "amuvarma/emilia-30k-TTS-iter"
new_dataset = new_dataset.shuffle(seed=42)
new_dataset.push_to_hub(push_name)
