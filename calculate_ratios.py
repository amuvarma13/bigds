from datasets import load_dataset
import os

# Load your dataset
dsn = "amuvarma/regconvos-kokoro"
push_name = "amuvarma/regconvos-kokoro-ratio"

ds = load_dataset(dsn, split="train")

num_proc = os.cpu_count() - 2

# Define a function to compute the ratio
def compute_ratio(example):
    # Calculate the length of the text and the length of the codes_list
    text_length = len(example['answer'])
    codes_length = len(example['codes_list'])
    # Avoid division by zero if codes_length is 0
    ratio = text_length / codes_length if codes_length > 0 else None
    return {'ratio': ratio}

# Map the function to add the new column
ds = ds.map(compute_ratio, num_proc=num_proc)

# Filter out examples with ratio bigger than 0.3 (and ensure ratio is not None)
ds = ds.filter(lambda x: x['ratio'] is not None and x['ratio'] <= 0.3)

# (Optional) Push the updated dataset to the hub
ds.push_to_hub(push_name)
