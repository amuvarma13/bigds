from datasets import Dataset, load_dataset
import os

num_proc = os.cpu_count() -2

dsn = "amuvarma/emilia-30k-TTS"
push_name = "amuvarma/emilia-30k-TTS-5g"

dataset = load_dataset(dsn, split='train')

# Remove columns not 'input_ids', 'attention_mask', 'labels'
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['input_ids']])

def merge_4_samples(examples):

    merged_input_ids = []

    
    merged_input_ids = sum(examples["input_ids"], [])
    
    if merged_input_ids > 8192:
        merged_input_ids = merged_input_ids[:8192]


    # Return them each as a *list of length 1* so map creates 1 new row, not 4.
    return {
        "input_ids": [merged_input_ids],
    }

# Map the dataset using a batch size of 4
merged_dataset = dataset.map(
    merge_4_samples,
    batched=True,
    batch_size=5,
    num_proc=num_proc,

)

merged_dataset = merged_dataset.shuffle(seed=42)
merged_dataset = merged_dataset.remove_columns([col for col in dataset.column_names if col not in ['input_ids']])

# Save the merged dataset
merged_dataset.push_to_hub(push_name)