from datasets import Dataset, load_dataset
import os

num_proc = os.cpu_count() -2

dsn = "amuvarma/text-messages-6m-processed-1"
push_name = "amuvarma/text-messages-6m-processed-combined"

dataset = load_dataset(dsn, split='train')

# Remove columns not 'input_ids', 'attention_mask', 'labels'
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']])

def merge_4_samples(examples):

    merged_input_ids = []
    merged_attention_mask = []
    merged_labels = []
    
    merged_input_ids = sum(examples["input_ids"], [])
    merged_attention_mask = sum(examples["attention_mask"], [])
    merged_labels = sum(examples["labels"], [])

    # Return them each as a *list of length 1* so map creates 1 new row, not 4.
    return {
        "input_ids": [merged_input_ids],
        "attention_mask": [merged_attention_mask],
        "labels": [merged_labels],
    }

# Map the dataset using a batch size of 4
merged_dataset = dataset.map(
    merge_4_samples,
    batched=True,
    batch_size=8,
    num_proc=num_proc,

)

merged_dataset = merged_dataset.shuffle(seed=42)
print(merged_dataset)

def truncate_sequences(dataset, max_length=8192):
    def truncate_row(example):
        if len(example['input_ids']) > max_length:
            example['input_ids'] = example['input_ids'][:max_length]
            example['attention_mask'] = example['attention_mask'][:max_length]
            example['labels'] = example['labels'][:max_length]
        return example
    
    return dataset.map(truncate_row, num_proc=num_proc)

truncated_dataset = truncate_sequences(merged_dataset)

# Save the merged dataset
truncated_dataset.push_to_hub(push_name)