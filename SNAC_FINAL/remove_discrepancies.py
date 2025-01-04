dsn = "amuvarma/snacced-flat-zuck-convo-sttsed-proc"
from datasets import load_dataset
dataset = load_dataset(dsn, split="train")

from datasets import load_dataset
import pandas as pd

# 2) Define the target token:
target_token = 128266 + (7 * 4096) + 1  # 156939

# 3) Use dataset.map() to compute the count of the target token for each example:
def count_target_token(example):
    # example["input_ids"] is a list of integers
    example["token_count"] = example["input_ids"].count(target_token)
    return example

dataset = dataset.map(count_target_token)

# 4) Convert to pandas DataFrame:
df = dataset.to_pandas()

# 5) Plot the histogram:
max_count = df["token_count"].max()

from datasets import load_dataset

target_token = 128266 + (7 * 4096) + 1  # 156939

# Step 1: Create a new column "audio_length" without storing large audio data
def store_audio_length(example):
    example["audio_length"] = len(example["audios"])
    return example

# remove_columns will remove the original "audios" from the dataset
dataset_small = dataset.map(
    store_audio_length, 
)

# Step 2: Compute the token_count in a second map
def count_target_token(example):
    example["token_count"] = example["input_ids"].count(target_token)
    return example

dataset_small = dataset_small.map(count_target_token)

# Step 3: Now filter for mismatches with minimal overhead
def has_mismatch(example):
    return example["token_count"] != example["audio_length"]

discrepancies = dataset_small.filter(has_mismatch)
print("Number of rows with discrepancies:", discrepancies.num_rows)

def count_target_token(example):
    target_token = 128266 + (7 * 4096) + 1
    example["token_count"] = example["input_ids"].count(target_token)
    return example

dataset = dataset.map(count_target_token)

# Now filter using the token_count column
def no_discrepancy(example):
    return example["token_count"] == len(example["audios"])

dataset_no_discrepancies = dataset.filter(no_discrepancy)

print("Original dataset size:", len(dataset))
print("Dataset size without discrepancies:", len(dataset_no_discrepancies))
