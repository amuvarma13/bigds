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

# Initialize the Audio feature (adjust parameters as needed)
audio_feature = Audio()

def process_audio(file_val):
    """
    Processes an audio file value and casts it into an Audio element.
    - If file_val is a dict containing "array" and "sampling_rate", it's already decoded.
    - Otherwise, if it's a string or dict with a "path", we decode it.
    In both cases, the decoded audio is passed through encode_example.
    """
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
    # Cast to Audio element
    return audio_feature.encode_example(decoded)

# -------------------------------
# Step 1. Add a top-level "speaker" column.
# (Assuming each row's json field has a key "speaker".)
dataset = dataset.map(lambda row: {"speaker": row["json"]["speaker"]})

# -------------------------------
# Step 2. Sort the dataset by speaker.
# This helps ensure that samples for a speaker are adjacent.
dataset = dataset.sort("speaker")

# -------------------------------
# Step 3. Use a batched map to create pairs.
def pair_samples(batch):
    """
    Expects a batch (a dict of lists) from the sorted dataset.
    Iterates through the batch and for any two consecutive rows with the same speaker,
    it creates one paired example.
    
    Returns a dict with four lists: audio_1, text_1, audio_2, text_2.
    """
    audio1, text1, audio2, text2 = [], [], [], []
    # We'll use these to remember the previous sample for a given speaker.
    current_speaker = None
    saved_sample = None
    
    # Loop over the batch
    for i in range(len(batch["speaker"])):
        speaker = batch["speaker"][i]
        current_audio = batch["mp3"][i]
        current_text = batch["json"][i]["text"]
        
        if speaker != current_speaker:
            # New speaker encountered: save this sample as the first sample.
            current_speaker = speaker
            saved_sample = (current_audio, current_text)
        else:
            # Same speaker as previous sample, so create a pair.
            try:
                processed_audio1 = process_audio(saved_sample[0])
                processed_audio2 = process_audio(current_audio)
            except Exception as e:
                print(f"Skipping speaker {speaker} due to audio processing error: {e}")
                # Reset to avoid reusing this speaker.
                current_speaker = None
                saved_sample = None
                continue
            audio1.append(processed_audio1)
            text1.append(saved_sample[1])
            audio2.append(processed_audio2)
            text2.append(current_text)
            # Reset after forming a pair so each speaker is used only once.
            current_speaker = None
            saved_sample = None
    return {"audio_1": audio1, "text_1": text1, "audio_2": audio2, "text_2": text2}

# Use batched mapping; adjust batch_size if needed to ensure speakers aren’t split across batches.
paired_dataset = dataset.map(pair_samples, batched=True, batch_size=10000)

# The resulting dataset will have columns "audio_1", "text_1", "audio_2", "text_2"
print(paired_dataset)
