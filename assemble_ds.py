from datasets import load_dataset, concatenate_datasets, Audio
from huggingface_hub import snapshot_download
import librosa
import os
import numpy as np

dsn = "amuvarma/brian-48k-KRVv68cDw2PBeOJypLrzaiI4kol2"

snapshot_download(
    repo_id=dsn,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

ds = load_dataset(dsn, split="train")

def process_row(row, idx):
    # Construct path to the wav file based on the row index
    wav_path = f"./wavs/{idx}.wav"
    
    # Check if the file exists
    if not os.path.exists(wav_path):
        # Add a flag to indicate this row should be filtered out later
        row["has_wav"] = False
        # Initialize with empty placeholder of the correct dtype
        row["enhanced_audio"] = {"array": np.zeros(0, dtype=np.float32), "sampling_rate": 16000}
        return row
    
    # Load audio using librosa with default sample rate and explicitly set dtype to float32
    audio_array, sampling_rate = librosa.load(wav_path, sr=None, dtype=np.float32)
    
    # Create enhanced_audio field with array and sampling rate
    row["enhanced_audio"] = {"array": audio_array, "sampling_rate": sampling_rate}
    row["has_wav"] = True
    
    return row

# Calculate number of processes to use (all cores minus 2)
num_cores = max(1, os.cpu_count() - 2)
print(f"Using {num_cores} processes for parallel processing")

# Apply the mapping function to the dataset with row indices and parallel processing
ds = ds.map(
    function=process_row,
    with_indices=True,
    num_proc=num_cores,
    desc="Processing audio files",
)

# Cast the enhanced_audio column to Audio type
ds = ds.cast_column("enhanced_audio", Audio())

# Filter the dataset to keep only rows with found WAV files
ds = ds.filter(lambda example: example["has_wav"])

# Remove the temporary has_wav column
ds = ds.remove_columns("has_wav")

# Now ds contains only rows with matching WAV files plus the enhanced_audio column
print(f"Dataset now has {len(ds)} rows with columns: {ds.column_names}")
print(f"Rows with missing WAV files were removed")

# Push the enhanced dataset to the Hugging Face Hub
ds = ds.push_to_hub("amuvarma/brian-48k-KRVv68cDw2PBeOJypLrzaiI4kol2-enhanced")