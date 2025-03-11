from datasets import Dataset, load_from_disk, concatenate_datasets
import os
import glob

# Repository name for the final dataset
repo_id = "amuvarma/grouped-smol-8192"

# Find all partition directories in the current directory
partition_dirs = sorted(glob.glob("processed_partition_*"))
print(f"Found {len(partition_dirs)} partition directories: {partition_dirs}")

if not partition_dirs:
    print("No partitions found. Make sure you're in the correct directory.")
    exit(1)

# Load each partition
print("Loading partitions from disk...")
partitions = []
for partition_dir in partition_dirs:
    try:
        print(f"Loading {partition_dir}...")
        partition = load_from_disk(partition_dir)
        print(f"  Loaded successfully: {len(partition)} examples")
        partitions.append(partition)
    except Exception as e:
        print(f"  Error loading {partition_dir}: {str(e)}")

# Check if we have any valid partitions
if not partitions:
    print("No valid partitions were loaded. Check the error messages above.")
    exit(1)

# Combine the partitions
print(f"Combining {len(partitions)} partitions...")
try:
    combined_dataset = concatenate_datasets(partitions)
    print(f"Combined dataset has {len(combined_dataset)} examples")
except Exception as e:
    print(f"Error combining partitions: {str(e)}")
    print("Attempting to upload individual partitions instead...")
    for i, partition in enumerate(partitions):
        try:
            partition_repo_id = f"{repo_id}-part-{i}"
            print(f"Uploading partition {i} to {partition_repo_id}...")
            partition.push_to_hub(partition_repo_id)
            print(f"  Successfully uploaded partition {i}")
        except Exception as e:
            print(f"  Error uploading partition {i}: {str(e)}")
    exit(1)

# Upload the combined dataset to Hugging Face Hub
print(f"Uploading combined dataset to {repo_id}...")
try:
    combined_dataset.push_to_hub(repo_id)
    print("Dataset uploaded successfully!")
except Exception as e:
    print(f"Error uploading combined dataset: {str(e)}")
    print("You may need to run this with more memory or try uploading individual partitions.")

print("Process completed")