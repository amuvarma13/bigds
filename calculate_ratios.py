from datasets import load_dataset
import os

# Load your dataset
dsn = "amuvarma/voice-assistant-adapted-1-100k-snacced"
ds = load_dataset(dsn, split="train")

num_proc = os.cpu_count() -2
# Define a function to compute the ratio
def compute_ratio(example):
    # Calculate the length of the text and the length of the codes_list
    text_length = len(example['text'])
    codes_length = len(example['codes_list'])
    # Avoid division by zero if codes_length is 0
    ratio = text_length / codes_length if codes_length > 0 else None
    return {'ratio': ratio}

# Map the function to add the new column
ds = ds.map(compute_ratio, num_proc=num_proc)

# (Optional) Push the updated dataset to the hub
push_name = "amuvarma/voice-assistant-adapted-1-100k-snacced-ratio"
ds.push_to_hub(push_name)
