from datasets import Dataset, load_dataset, concatenate_datasets
from itertools import chain, islice
import time
import math

# Dataset parameters
dsn = "amuvarma/emilia-snac-merged-18m-smol-TTS"
total_expected_examples = 17114684  # You already know this number
partition_size = 570489  # Number of examples per partition
chunk_size = 8192  # Size of each token chunk
num_partitions = math.ceil(total_expected_examples / partition_size)

# Load the dataset in streaming mode
dataset = load_dataset(dsn, streaming=True)
train_dataset = dataset['train']

# If starting from a later partition, we need to skip examples
start_partition = 13  # Start from partition 15
if start_partition > 0:
    examples_to_skip = start_partition * partition_size
    print(f"Skipping first {examples_to_skip} examples to start from partition {start_partition}")
    # Skip examples by iterating through them but not processing them
    skip_count = 0
    for _ in islice(train_dataset, examples_to_skip):
        skip_count += 1
        if skip_count % 10000 == 0:
            print(f"  Skipped {skip_count}/{examples_to_skip} examples...")
    print(f"  Finished skipping {skip_count} examples. Now starting from partition {start_partition}")

# Filter to keep only input_ids
# First, peek at a sample to get column names
sample = next(iter(train_dataset))
all_columns = list(sample.keys())
columns_to_remove = [col for col in all_columns if col != 'input_ids']
filtered_dataset = train_dataset.remove_columns(columns_to_remove)

print(f"Number of partitions to process: {num_partitions}")
processed_partitions = []

# Set the starting partition index
start_partition = 15  # Start from partition 15

# Process dataset in partitions, starting from the specified index
for i in range(start_partition, num_partitions):
    print(f"Processing partition {i+1}/{num_partitions}")
    start_time = time.time()
    
    # Use islice to get a partition of the streaming dataset
    partition_iterator = islice(filtered_dataset, partition_size)
    
    # Collect input_ids from this partition
    all_tokens = []
    count = 0
    
    for example in partition_iterator:
        all_tokens.extend(example['input_ids'])
        count += 1
        if count % 10000 == 0:
            print(f"  Processed {count} examples in current partition")
    
    if count == 0:
        print(f"  No more examples to process in partition {i+1}")
        break
        
    print(f"  Collected {len(all_tokens)} tokens from {count} examples")
    
    # Create chunks of the specified size
    num_chunks = len(all_tokens) // chunk_size
    chunks = [all_tokens[j*chunk_size:(j+1)*chunk_size] for j in range(num_chunks)]
    print(f"  Created {len(chunks)} chunks of size {chunk_size}")
    
    # Create a dataset from these chunks
    new_dataset = Dataset.from_dict({"input_ids": chunks})
    
    # Add to our list of processed partitions
    processed_partitions.append(new_dataset)
    
    end_time = time.time()
    print(f"  Time taken for partition {i+1}: {end_time - start_time:.2f} seconds")
    
    # Optional: Push this partition to hub if you don't want to combine later
    # push_name = f"{account_name}/emilia-snac-merged-18m-smol-TTS-8192-part-{i+1}-of-{num_partitions}"
    # new_dataset.push_to_hub(push_name)

# Combine all partitions
if processed_partitions:
    print("Combining all partitions...")
    combined_dataset = concatenate_datasets(processed_partitions)
    print(f"Combined dataset has {len(combined_dataset)} examples")
    
    # Push to hub
    combined_dataset.push_to_hub(f"amuvarma/emilia-snac-merged-18m-smol-TTS-8192")
    print("Dataset uploaded successfully!")
else:
    print("No partitions were processed. Check your dataset.")