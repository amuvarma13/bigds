from datasets import load_dataset

from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amphion/Emilia-Dataset"
path = "Emilia/EN/*.tar"
dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")

dataset = dataset.select(range(10))

import os
from datasets import Dataset, Audio
from tqdm import tqdm

# Initialize the Audio feature (adjust sampling_rate if needed)
audio_feature = Audio()

def decode_audio(file_val):

    if isinstance(file_val, dict):
        # Use pre-decoded audio if available.
        if "array" in file_val and "sampling_rate" in file_val:
            return file_val
        elif "path" in file_val:
            path = file_val["path"]
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} not found.")
            file_dict = file_val.copy()
            if "bytes" not in file_dict:
                file_dict["bytes"] = None
            return audio_feature.decode_example(file_dict)
        else:
            raise ValueError("Dictionary does not contain 'array' or 'path'.")
    elif isinstance(file_val, str):
        if not os.path.exists(file_val):
            raise FileNotFoundError(f"File {file_val} not found.")
        file_dict = {"path": file_val, "bytes": None}
        return audio_feature.decode_example(file_dict)
    else:
        raise ValueError("Unsupported type for audio file")

def pair_generator(dataset):

    unmatched = {}
    for row in tqdm(dataset, total=len(dataset)):
        speaker = row["json"]["speaker"]
        try:
            current_audio = decode_audio(row["mp3"])
        except Exception as e:
            print(f"Skipping sample for speaker {speaker} due to error: {e}")
            continue

        if speaker in unmatched:
            prev_row = unmatched.pop(speaker)
            try:
                previous_audio = decode_audio(prev_row["mp3"])
            except Exception as e:
                print(f"Skipping pair for speaker {speaker} due to error: {e}")
                continue
            yield {
                "audio_1": previous_audio,
                "text_1": prev_row["json"]["text"],
                "audio_2": current_audio,
                "text_2": row["json"]["text"]
            }
        else:
            unmatched[speaker] = row

# Create the new dataset from the generator.
paired_dataset = Dataset.from_generator(lambda: pair_generator(dataset))

print(paired_dataset)

paired_dataset = paired_dataset.push_to_hub("amuvarma/Emilia-Dataset-paired-2")
