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
        # Return None to indicate this row should be filtered out
        return None
    
    # Load audio using librosa with default sample rate
    audio_array, sampling_rate = librosa.load(wav_path, sr=None)
    
    # Create enhanced_audio field with array and sampling rate
    row["enhanced_audio"] = {"array": audio_array, "sampling_rate": sampling_rate}
    
    return row

# Calculate number of processes to use (all cores minus 2)
num_cores = max(1, os.cpu_count() - 2)
print(f"Using {num_cores} processes for parallel processing")

# Apply the mapping function to the dataset with row indices and parallel processing
# The filter_function=None parameter ensures rows that return None are filtered out
ds = ds.map(
    function=process_row,
    with_indices=True,
    num_proc=num_cores,
    remove_columns=None,  # Keep all original columns
    desc="Processing audio files",
    filter_function=lambda x: x is not None  # Filter out None returns (missing WAVs)
)

# Cast the enhanced_audio column to Audio type
ds = ds.cast_column("enhanced_audio", Audio())

# Now ds contains only rows with matching WAV files plus the enhanced_audio column
print(f"Dataset now has {len(ds)} rows with columns: {ds.column_names}")
print(f"Rows with missing WAV files were removed")

ds = ds.push_to_hub("amuvarma/brian-48k-KRVv68cDw2PBeOJypLrzaiI4kol2-enhanced")

# Now ds contains the original data plus the enhanced_audio column
print(f"Dataset now has {len(ds)} rows with columns: {ds.column_names}")