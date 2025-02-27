from datasets import load_dataset

from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amphion/Emilia-Dataset"
path = "Emilia/EN/*.tar"
dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")

subdataset = dataset.select(range(10))

import json

def transform_dataset(example):
    """
    Transform a dataset example to have text and audio columns.
    
    Args:
        example: Dictionary with mp3 and json fields
        
    Returns:
        Dictionary with text and audio fields
    """
    # Parse the JSON field if it's a string
    if isinstance(example["json"], str):
        json_data = json.loads(example["json"])
    else:
        json_data = example["json"]
    
    # Extract the text from the JSON data
    text = json_data["text"]
    
    # Return a new structure with just text and audio
    return {
        "text": text,
        "audio": example["mp3"]  # Keep the original audio structure
    }

# Usage:
transformed_dataset = subdataset.map(transform_dataset)

print(transformed_dataset)
