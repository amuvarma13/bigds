from datasets import Dataset, load_dataset

# For illustration, let's suppose `dataset` is already loaded:
# dataset = load_dataset(...)["train"]
# and it has columns: 'input_ids', 'attention_mask', 'labels', each of which is a list of ints.

dsn = "amuvarma/zuck-nopunc-wcodes-TTTTS"
push_name = "amuvarma/zuck-nopunc-wcodes-TTTTS-merged"

dataset = load_dataset(dsn, split='train')

def merge_4_samples(examples):

    merged_input_ids = []
    merged_attention_mask = []
    merged_labels = []
    
    # We can simply use Python's sum over lists to flatten:
    # sum(list_of_lists, []) concatenates them.
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
    batch_size=4,
    # optionally remove the old columns if you only want the merged ones:
    # remove_columns=dataset.column_names
)

print(merged_dataset)

# Save the merged dataset
merged_dataset.push_to_hub(push_name)