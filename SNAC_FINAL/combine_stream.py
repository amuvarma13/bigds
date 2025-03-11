from datasets import Dataset, load_dataset, concatenate_datasets
from itertools import chain, islice
import time
import math
import traceback

# Dataset parameters
dsn = "amuvarma/emilia-snac-merged-18m-smol-TTS"
total_expected_examples = 17114684
partition_size = 570489
chunk_size = 8192
num_partitions = math.ceil(total_expected_examples / partition_size)
start_partition = 0

print(f"Starting dataset processing: {dsn}")
print(f"Total partitions: {num_partitions}, Chunk size: {chunk_size}")

try:
    # Load the dataset in streaming mode
    dataset = load_dataset(dsn, streaming=True)
    train_dataset = dataset['train']
    
    # Get column names from a sample
    sample = next(iter(train_dataset))
    all_columns = list(sample.keys())
    columns_to_remove = [col for col in all_columns if col != 'input_ids']
    filtered_dataset = train_dataset.remove_columns(columns_to_remove)
    
    processed_partitions = []
    
    # Process dataset in partitions
    for i in range(start_partition, num_partitions):
        try:
            print(f"\n== Processing partition {i+1}/{num_partitions} ==")
            start_time = time.time()
            
            # Get a batch of examples
            partition_iterator = islice(filtered_dataset, partition_size)
            
            # Collect tokens
            all_tokens = []
            count = 0
            
            for example in partition_iterator:
                all_tokens.extend(example['input_ids'])
                count += 1
                if count % 10000 == 0:
                    print(f"  Processed {count} examples")
            
            if count == 0:
                print(f"  No more examples available. Reached end of dataset.")
                break
                
            print(f"  Collected {len(all_tokens)} tokens from {count} examples")
            
            # Create chunks
            num_chunks = len(all_tokens) // chunk_size
            chunks = [all_tokens[j*chunk_size:(j+1)*chunk_size] for j in range(num_chunks)]
            print(f"  Created {len(chunks)} chunks")
            
            # Create dataset
            new_dataset = Dataset.from_dict({"input_ids": chunks})
            processed_partitions.append(new_dataset)
            
            end_time = time.time()
            print(f"  Time: {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"ERROR in partition {i+1}: {str(e)}")
            print(f"Skipping to next partition...")
            continue
    
    # Combine all partitions
    if processed_partitions:
        try:
            print("\nCombining all partitions...")
            combined_dataset = concatenate_datasets(processed_partitions)
            print(f"Combined dataset has {len(combined_dataset)} examples")
            
            print(f"Pushing dataset to hub...")
            combined_dataset.push_to_hub(f"amuvarma/emilia-snac-merged-18m-smol-TTS-8192")
            print("Upload successful!")
        except Exception as e:
            print(f"ERROR during upload: {str(e)}")
            print("Attempting to save locally...")
            for idx, partition in enumerate(processed_partitions):
                try:
                    partition.save_to_disk(f"processed_partition_{idx}")
                    print(f"Saved partition {idx} to disk")
                except:
                    print(f"Failed to save partition {idx}")
    else:
        print("No partitions were processed successfully.")

except Exception as e:
    print(f"CRITICAL ERROR: {str(e)}")
    print("Process terminated")

print("Job completed")