from datasets import load_dataset

from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amphion/Emilia-Dataset"
path = "Emilia/EN/*.tar"
dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")

dataset = dataset.select(range(10))
print(dataset[0]["json"])

import os
from datasets import Dataset, Audio
import datasets

from collections import defaultdict

def combine_by_speaker(batch):
    """
    Expects a batch with keys:
      - "mp3": a list of dictionaries, each with keys like "path", "array", "sampling_rate"
      - "json": a dictionary with keys "text" and "speaker", where each value is a list.
    
    For each speaker that appears exactly twice in the batch, this function outputs a new row
    with two mp3 entries, under keys "mp3_1" and "mp3_2".
    """
    # Create a mapping from speaker to the indices where it appears
    speaker_to_indices = defaultdict(list)
    for idx, speaker in enumerate(batch["json"]["speaker"]):
        speaker_to_indices[speaker].append(idx)
    
    # Prepare output lists
    new_mp3_1, new_mp3_2 = [], []
    
    # For each speaker with exactly 2 entries, combine the corresponding mp3s.
    for speaker, indices in speaker_to_indices.items():
        if len(indices) == 2:
            i1, i2 = indices
            new_mp3_1.append(batch["mp3"][i1])
            new_mp3_2.append(batch["mp3"][i2])
    
    return {"mp3_1": new_mp3_1, "mp3_2": new_mp3_2}

# Apply the map function over the entire dataset.
# (Using batch_size=len(dataset) ensures that all rows are in one batch,
#  which is important if rows for the same speaker might be in different batches.)
new_dataset = dataset.map(combine_by_speaker, batched=True, batch_size=len(dataset))

print(new_dataset)



