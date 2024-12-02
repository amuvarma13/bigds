from datasets import load_dataset
import datasets
import numpy as np
import torch
from tqdm.auto import tqdm


ds_name = "amuvarma/100k-fac_0"
dsy = load_dataset(ds_name)

batch_size=200


ds = dsy["train"]

print("dataset loaded")

def remove_consecutive_duplicates_batched(ds, batch_size=1000):
    # Get the total number of batches
    num_batches = len(ds) // batch_size + (1 if len(ds) % batch_size != 0 else 0)

    # Create a progress bar
    progress_bar = tqdm(total=num_batches, desc="Processing batches", unit="batch")

    def process_batch(batch, update_progress):
        columns = ['facodec_0', 'facodec_1', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5']
        
        for i in range(len(batch['facodec_1'])):
            # Create a boolean mask for elements in facodec_1 that are different from their previous element
            mask = [True] + [batch['facodec_1'][i][j] != batch['facodec_1'][i][j-1] for j in range(1, len(batch['facodec_1'][i]))]
            
            # Apply the mask to all columns
            for col in columns:
                batch[col][i] = [x for x, keep in zip(batch[col][i], mask) if keep]
        
        # Update the progress bar
        update_progress(1)
        
        return batch

    def update_progress(num):
        progress_bar.update(num)

    # Apply the mapping with progress bar
    ds = ds.map(
        process_batch,
        fn_kwargs={'update_progress': update_progress},
        batched=True,
        batch_size=batch_size
    )

    # Close the progress bar
    progress_bar.close()

    return ds

# Usage
ds = remove_consecutive_duplicates_batched(ds)

print("duplicates removed")


def add_offset_to_facodec_batched(ds, offset=128259, batch_size=32):
    # Get the total number of batches
    num_batches = len(ds) // batch_size + (1 if len(ds) % batch_size != 0 else 0)

    # Create a progress bar
    progress_bar = tqdm(total=num_batches, desc="Processing batches", unit="batch")

    def process_batch(batch, update_progress):
        for col in ['facodec_0', 'facodec_1', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5']:
            # Process each list in the batch separately
            processed_lists = []
            for lst in batch[col]:
                # Add offset to all elements in the list
                processed_lists.append([x + offset for x in lst])
            
            # Update the batch with the processed lists
            batch[col] = processed_lists
        
        # Create facodec_1_shifted column
        batch["facodec_1_shifted"] = batch["facodec_1"]
        
        # Update the progress bar
        update_progress(1)
        
        return batch

    def update_progress(num):
        progress_bar.update(num)

    # Apply the mapping with progress bar
    ds = ds.map(
        process_batch,
        fn_kwargs={'update_progress': update_progress},
        batched=True,
        batch_size=batch_size
    )

    # Close the progress bar
    progress_bar.close()

    return ds

# Usage
ds = add_offset_to_facodec_batched(ds)

print("offset added")


max_length = 2000


def prepare_dataset_for_model_batched(ds, batch_size=32):
    # First, find the maximum length across all rows
    max_length = max(len(row['facodec_0']) for row in ds)

    # Calculate the total number of batches
    num_batches = len(ds) // batch_size + (1 if len(ds) % batch_size != 0 else 0)

    # Create a progress bar
    progress_bar = tqdm(total=num_batches, desc="Preparing dataset", unit="batch")

    def process_batch(batch, update_progress):
        # Get original lengths
        original_lengths = [len(row) for row in batch['facodec_0']]

        # Create padded arrays for all facodec columns
        for i in range(6):
            col_name = f'facodec_{i}'
            batch[col_name] = pad_sequences(batch[col_name], max_length).tolist()

        # Create padded array for facodec_1_shifted
        facodec_1_shifted = pad_sequences(batch['facodec_1_shifted'], max_length)

        # Create attention mask
        attention_mask = np.zeros((len(batch['facodec_0']), max_length), dtype=np.int64)
        for i, length in enumerate(original_lengths):
            attention_mask[i, :length] = 1

        # Shift facodec_1_shifted and append 256002
        for i, length in enumerate(original_lengths):
            facodec_1_shifted[i] = np.roll(facodec_1_shifted[i], -1)
            facodec_1_shifted[i, length - 1] = 256002

        # Update batch with processed arrays
        batch['input_ids'] = batch['facodec_0']
        batch['attention_mask'] = attention_mask.tolist()
        batch['facodec_1_shifted'] = facodec_1_shifted.tolist()

        # Update the progress bar
        update_progress(1)

        return batch

    def pad_sequences(sequences, max_len):
        return np.array([np.pad(seq, (0, max_len - len(seq)), 'constant') for seq in sequences])

    def update_progress(num):
        progress_bar.update(num)

    # Apply the processing to each batch with progress bar
    ds = ds.map(
        process_batch,
        fn_kwargs={'update_progress': update_progress},
        batched=True,
        batch_size=batch_size
    )

    # Close the progress bar
    progress_bar.close()

    # Set the format of the dataset to convert lists to pytorch tensors
    ds = ds.with_format("torch")

    return ds

# Usage
ds = prepare_dataset_for_model_batched(ds)

print("dataset prepared")
ds.push_to_hub("amuvarma/multilayer-100k-0-dedup-llama")