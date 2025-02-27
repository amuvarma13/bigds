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

# Initialize the Audio feature (set parameters as needed)
audio_feature = Audio()

def process_audio(file_val):
    """
    Processes an audio file value and casts it into an Audio element.

    - If file_val is a dict with "array" and "sampling_rate", it is assumed to be pre-decoded.
    - If file_val has a "path" (or is a string), it will be decoded.
    - The resulting decoded audio is then passed through encode_example so that it is cast as an Audio element.
    """
    # If it's a dict with pre-decoded audio, use it directly.
    if isinstance(file_val, dict):
        if "array" in file_val and "sampling_rate" in file_val:
            decoded = file_val
        elif "path" in file_val:
            path = file_val["path"]
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} not found.")
            file_dict = file_val.copy()
            if "bytes" not in file_dict:
                file_dict["bytes"] = None
            decoded = audio_feature.decode_example(file_dict)
        else:
            raise ValueError("Dictionary must have either 'array' & 'sampling_rate' or 'path'.")
    elif isinstance(file_val, str):
        # Treat string as a file path.
        if not os.path.exists(file_val):
            raise FileNotFoundError(f"File {file_val} not found.")
        decoded = audio_feature.decode_example({"path": file_val, "bytes": None})
    else:
        raise ValueError("Unsupported type for audio file")

    # Cast the decoded audio (a dict with 'array' and 'sampling_rate') into an Audio element.
    return audio_feature.encode_example(decoded)

def pair_generator(dataset):
    """
    Iterates over the dataset and yields paired examples (audio_1, text_1, audio_2, text_2)
    for speakers that have at least two valid samples.
    """
    unmatched = {}
    for row in tqdm(dataset, total=len(dataset)):
        speaker = row["json"]["speaker"]
        try:
            current_audio = process_audio(row["mp3"])
        except Exception as e:
            print(f"Skipping sample for speaker {speaker} due to error: {e}")
            continue

        if speaker in unmatched:
            prev_row = unmatched.pop(speaker)
            try:
                previous_audio = process_audio(prev_row["mp3"])
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

# Create the new dataset using the generator.
paired_dataset = Dataset.from_generator(lambda: pair_generator(dataset))

print(paired_dataset)





paired_dataset = paired_dataset.push_to_hub("amuvarma/Emilia-Dataset-p2")
