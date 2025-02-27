from datasets import load_dataset

from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amphion/Emilia-Dataset"
path = "Emilia/EN/*.tar"
dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")

dataset = dataset.select(range(10))

import os
from datasets import Dataset, Audio
import datasets

# Initialize the Audio feature (adjust parameters if needed)
audio_feature = Audio()

def process_audio(file_val):
    """
    Processes an audio file value and casts it into an Audio element.
    - If file_val is a dict with "array" and "sampling_rate", it is assumed to be pre-decoded.
    - Otherwise, if file_val is a string or dict with a "path", it is decoded via the Audio feature.
    The decoded audio is then encoded to an Audio element.
    """
    # Use pre-decoded audio if available.
    if isinstance(file_val, dict) and "array" in file_val and "sampling_rate" in file_val:
        decoded = file_val
    else:
        if isinstance(file_val, str):
            file_dict = {"path": file_val, "bytes": None}
        elif isinstance(file_val, dict) and "path" in file_val:
            file_dict = file_val.copy()
            file_dict.setdefault("bytes", None)
        else:
            raise ValueError("Unexpected audio format")
        if not os.path.exists(file_dict["path"]):
            raise FileNotFoundError(f"File {file_dict['path']} not found.")
        decoded = audio_feature.decode_example(file_dict)
    # Encode the decoded audio into an Audio element.
    return audio_feature.encode_example(decoded)

def pair_samples(batch):
    """
    Expects a batch (a dict of lists) from a sorted dataset.
    Iterates through the batch and for any two consecutive examples with the same speaker,
    creates a paired example. Returns a list of dictionaries (each dict is one new example).
    
    Note: This approach will only pair examples that occur consecutively within the batch.
    """
    pairs = []
    # Loop over indices in the batch (stop before the last element)
    i = 0
    while i < len(batch["speaker"]) - 1:
        # Check if the current and next example belong to the same speaker.
        if batch["speaker"][i] == batch["speaker"][i + 1]:
            try:
                audio_1 = process_audio(batch["mp3"][i])
                audio_2 = process_audio(batch["mp3"][i + 1])
            except Exception as e:
                print(f"Skipping speaker {batch['speaker'][i]} due to audio processing error: {e}")
                i += 1
                continue
            pair = {
                "audio_1": audio_1,
                "text_1": batch["json"][i]["text"],
                "audio_2": audio_2,
                "text_2": batch["json"][i + 1]["text"]
            }
            pairs.append(pair)
            # Skip the next example since it's been paired
            i += 2
        else:
            i += 1
    return pairs

# Assume your original dataset is loaded as `dataset`
# For example: dataset = load_dataset("your_dataset_name", split="train")

# Step 1: Add a top-level "speaker" column (extracted from the nested json field)
dataset = dataset.map(lambda row: {"speaker": row["json"]["speaker"]})

# Step 2: Sort the dataset by "speaker" to help group same-speaker examples together.
dataset = dataset.sort("speaker")

# Step 3: Use a batched map to create pairs.
# Here we return a list of new examples (each as a dict), so the output length can vary.
paired_dataset = dataset.map(pair_samples, batched=True, batch_size=10000)

# At this point, paired_dataset contains the new examples with columns:
# "audio_1", "text_1", "audio_2", "text_2"
print(paired_dataset)
