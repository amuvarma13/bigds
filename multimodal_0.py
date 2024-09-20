from datasets import load_dataset
import datasets
import numpy as np
import torch


ds_name = "amuvarma/1m-fac_0"
dsy = load_dataset(ds_name)


ds = dsy["train"]



def remove_consecutive_duplicates_batched(ds, batch_size=32):
    def process_batch(batch):
        for col in ['facodec_0', 'facodec_1', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5']:
            # Convert list of lists to 2D numpy array
            arr = np.array(batch[col].tolist())
            
            # Create a boolean mask for elements that are different from their previous element
            mask = np.ones(arr.shape, dtype=bool)
            mask[:, 1:] = arr[:, 1:] != arr[:, :-1]
            
            # Apply the mask to each row
            batch[col] = [row[m].tolist() for row, m in zip(arr, mask)]
        
        return batch

    return ds.map(process_batch, batched=True, batch_size=batch_size)

# Usage
ds = remove_consecutive_duplicates_batched(ds)



def add_offset_to_facodec_batched(ds, offset=256003, batch_size=32):
    def process_batch(batch):
        for col in ['facodec_0', 'facodec_1', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5']:
            # Convert list of lists to 2D numpy array
            arr = np.array(batch[col].tolist())
            
            # Add offset to all elements
            arr += offset
            
            # Convert back to list of lists
            batch[col] = arr.tolist()
        
        # Create facodec_1_shifted column
        batch["facodec_1_shifted"] = batch["facodec_1"]
        
        return batch

    return ds.map(process_batch, batched=True, batch_size=batch_size)

# Usage
ds = add_offset_to_facodec_batched(ds)

max_length = 2000


def prepare_dataset_for_model_batched(ds, batch_size=32):
    # First, find the maximum length across all rows
    max_length = max(len(row['facodec_0']) for row in ds)

    def process_batch(batch):
        # Convert lists to numpy arrays for efficient processing
        input_ids = np.array(batch['facodec_0'].tolist())
        facodec_1 = np.array(batch['facodec_1'].tolist())
        facodec_1_shifted = np.array(batch['facodec_1_shifted'].tolist())

        # Get original lengths
        original_lengths = np.array([len(row) for row in input_ids])

        # Create attention mask
        attention_mask = np.zeros((len(input_ids), max_length), dtype=np.int64)
        for i, length in enumerate(original_lengths):
            attention_mask[i, :length] = 1

        # Shift facodec_1_shifted and append 256002
        facodec_1_shifted = np.roll(facodec_1_shifted, -1, axis=1)
        facodec_1_shifted[np.arange(len(facodec_1_shifted)), original_lengths - 1] = 256002

        # Pad arrays
        input_ids = pad_2d_array(input_ids, max_length)
        facodec_1 = pad_2d_array(facodec_1, max_length)
        facodec_1_shifted = pad_2d_array(facodec_1_shifted, max_length)

        # Update batch with processed arrays
        batch['input_ids'] = input_ids.tolist()
        batch['attention_mask'] = attention_mask.tolist()
        batch['facodec_1'] = facodec_1.tolist()
        batch['facodec_1_shifted'] = facodec_1_shifted.tolist()

        return batch

    def pad_2d_array(arr, target_length):
        pad_width = ((0, 0), (0, target_length - arr.shape[1]))
        return np.pad(arr, pad_width, mode='constant', constant_values=0)

    # Apply the processing to each batch
    ds = ds.map(process_batch, batched=True, batch_size=batch_size)

    # Set the format of the dataset to convert lists to pytorch tensors
    ds = ds.with_format("torch")

    return ds

# Usage
ds = prepare_dataset_for_model_batched(ds)

ds.push_to_hub("amuvarma/multilayer-1m-0")