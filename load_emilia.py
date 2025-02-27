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
import datasets  # Ensure you have the datasets library installed

# Initialize the Audio feature (set any parameters as needed)
audio_feature = Audio()

def process_audio(file_val):
    """
    Processes an audio file value and casts it into an Audio element.
    
    - If file_val is a dict with "array" and "sampling_rate", it is assumed to be pre-decoded.
    - Otherwise, if file_val is a string or a dict with a "path", it is decoded via the Audio feature.
    - Finally, the decoded audio is encoded using audio_feature.encode_example to cast it as an Audio element.
    """
    if isinstance(file_val, dict) and "array" in file_val and "sampling_rate" in file_val:
        # Pre-decoded audio
        decoded = file_val
    else:
        # If file_val is a string or dict with a "path", prepare it for decoding.
        if isinstance(file_val, str):
            file_dict = {"path": file_val, "bytes": None}
        elif isinstance(file_val, dict) and "path" in file_val:
            file_dict = file_val.copy()
            file_dict.setdefault("bytes", None)
        else:
            raise ValueError("Unexpected audio format")
        # Optionally, check that the file exists on disk.
        if not os.path.exists(file_dict["path"]):
            raise FileNotFoundError(f"File {file_dict['path']} not found.")
        decoded = audio_feature.decode_example(file_dict)
    
    # Cast the decoded audio (a dict with "array" and "sampling_rate") into an Audio element.
    return audio_feature.encode_example(decoded)

def extract_pair(group):
    """
    For a given grouped example (by speaker), extracts a pair of samples (if at least 2 exist).
    The group is a dict with lists for each field.
    """
    if len(group["mp3"]) < 2:
        # Return an empty dict if fewer than 2 samples are present.
        return {}
    try:
        audio_1 = process_audio(group["mp3"][0])
        audio_2 = process_audio(group["mp3"][1])
    except Exception as e:
        print(f"Skipping group for speaker {group['speaker'][0]} due to error: {e}")
        return {}
    return {
        "audio_1": audio_1,
        "text_1": group["json"][0]["text"],
        "audio_2": audio_2,
        "text_2": group["json"][1]["text"]
    }

def has_two_samples(group):
    """Keeps groups with at least 2 samples."""
    return len(group["mp3"]) >= 2

# Assume your original dataset is loaded as `dataset`
# For example: dataset = load_dataset("your_dataset_name", split="train")

# 1. Add a top-level "speaker" column extracted from the nested json field.
dataset = dataset.map(lambda row: {"speaker": row["json"]["speaker"]})

# 2. Group the dataset by the "speaker" column.
grouped_dataset = dataset.group_by("speaker", keep_in_memory=True)

# 3. Filter out speakers with fewer than 2 samples.
grouped_dataset = grouped_dataset.filter(has_two_samples)

# 4. Map over each group to extract a pair.
paired_dataset = grouped_dataset.map(extract_pair, batched=False)

# 5. Remove unnecessary columns (keep only the four desired ones).
columns_to_keep = ["audio_1", "text_1", "audio_2", "text_2"]
paired_dataset = paired_dataset.remove_columns([col for col in paired_dataset.column_names if col not in columns_to_keep])

print(paired_dataset)
