from datasets import load_dataset

from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amphion/Emilia-Dataset"
path = "Emilia/EN/*.tar"
dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")

dataset = dataset.select(range(10))
print(dataset[0]["json"]["speaker"])

import os
from datasets import Dataset, Audio
import datasets

from collections import defaultdict
def combine_same_speakers(examples):
    """
    Map function that combines pairs of rows with the same speaker ID.
    
    Args:
        examples: A batch of examples from the dataset
        
    Returns:
        Dictionary with combined examples
    """
    # Initialize result dictionary with new structure
    result = {
        "mp3_1": [],
        "mp3_2": [],
        "json": []
    }
    
    # Group examples by speaker ID
    speaker_groups = {}
    for i in range(len(examples["json"])):
        speaker = examples["json"][i]["speaker"]
        if speaker not in speaker_groups:
            speaker_groups[speaker] = []
        speaker_groups[speaker].append(i)
    
    # Combine rows that have exactly 2 examples with the same speaker
    for speaker, indices in speaker_groups.items():
        if len(indices) == 2:
            idx1, idx2 = indices
            result["mp3_1"].append(examples["mp3"][idx1])
            result["mp3_2"].append(examples["mp3"][idx2])
            result["json"].append({
                "speaker": speaker,
                "text_1": examples["json"][idx1]["text"],
                "text_2": examples["json"][idx2]["text"]
            })
    
    return result
# Process the entire dataset as a single batch
combined_dataset = dataset.map(
    combine_same_speakers, 
    batched=True,
    batch_size=len(dataset)  # Important: use the full dataset size as batch size
)