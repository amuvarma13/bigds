import os
import time
from itertools import chain
import concurrent.futures

from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download

# Define dataset parameters
dsn = "amuvarma/text-messages-6m-iids"
partition_size = 520093
chunk_size = 1024

# Download the dataset snapshot
snapshot_download(
    repo_id=dsn,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

# Load the dataset
dataset = load_dataset(dsn, split='train')
print(dataset)

# Use the actual dataset length
dataset_length = len(dataset)

# Remove any columns that are not 'input_ids'
dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'input_ids'])

# Calculate the number of partitions
num_partitions = dataset_length // partition_size
print(f"Number of partitions: {num_partitions}")

def process_partition(i):
    """Process a partition of the dataset:
       - Select the partition range.
       - Flatten the list of input_ids.
       - Chunk the tokens.
       - Create a new dataset from chunks.
       - Push the new dataset to the hub.
    """
    print(f"Processing partition {i}")
    start = i * partition_size
    end = (i + 1) * partition_size 
    partition = dataset.select(range(start, end))
    
    start_time = time.time()
    # Flatten the list of tokens from the partition
    all_tokens = list(chain.from_iterable(partition["input_ids"]))
    
    # Determine the number of full chunks we can form
    num_chunks = len(all_tokens) // chunk_size  # dropping any leftover tokens
    chunks = [all_tokens[j * chunk_size:(j + 1) * chunk_size] for j in range(num_chunks)]
    
    # Create a new dataset from the chunks
    new_dataset = Dataset.from_dict({"input_ids": chunks})
    
    end_time = time.time()
    print(f"Time taken for partition {i}: {end_time - start_time:.2f} seconds")
    
    # Push this partition's dataset to the hub
    push_name = f"{dsn}-{i}-of-{num_partitions}"
    new_dataset.push_to_hub(push_name)
    
    return new_dataset

# Process partitions in parallel using a ThreadPoolExecutor
processed_partitions = []
max_workers = 20  # Adjust this number based on your system's capabilities

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_partition, i) for i in range(num_partitions)]
    for future in concurrent.futures.as_completed(futures):
        processed_partitions.append(future.result())

# Concatenate all processed partitions into a single dataset and push to hub
combined_dataset = concatenate_datasets(processed_partitions)
combined_dataset.push_to_hub(f"{dsn}-grouped-{chunk_size}")
