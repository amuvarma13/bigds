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
from datasets import Dataset

def combine_same_speakers(dataset):
    """
    Process a dataset to combine rows with the same speaker ID.
    
    Args:
        dataset: A HuggingFace dataset
        
    Returns:
        A new Dataset with the combined pairs
    """
    # Convert to standard Python lists/dicts for easier processing
    all_examples = list(dataset)
    
    # Group by speaker
    speaker_groups = {}
    for i, example in enumerate(all_examples):
        speaker = example["json"]["speaker"]
        if speaker not in speaker_groups:
            speaker_groups[speaker] = []
        speaker_groups[speaker].append(i)
    
    # Prepare new data structure
    combined_data = {
        "mp3_1": [],
        "mp3_2": [],
        "text_1": [],
        "text_2": [],
        "speaker": []
    }
    
    # Only process speakers with exactly 2 examples
    for speaker, indices in speaker_groups.items():
        if len(indices) == 2:
            idx1, idx2 = indices
            combined_data["mp3_1"].append(all_examples[idx1]["mp3"])
            combined_data["mp3_2"].append(all_examples[idx2]["mp3"])
            combined_data["text_1"].append(all_examples[idx1]["json"]["text"])
            combined_data["text_2"].append(all_examples[idx2]["json"]["text"])
            combined_data["speaker"].append(speaker)
    
    # Create a new Dataset
    return Dataset.from_dict(combined_data)

# Usage:
# combined_dataset = combine_same_speakers(dataset)
combined_dataset = dataset.map(
    combine_same_speakers, 
    batched=True,
    batch_size=len(dataset)  # Important: use the full dataset size as batch size
)