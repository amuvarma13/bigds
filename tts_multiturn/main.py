import numpy as np
from text_to_speech import generate_and_resample_speech
import wave
from datasets import load_dataset, Audio

def text_to_audio(text):
    if text is None:
        return None
    resampled_audio, sample_rate = generate_and_resample_speech(text)
    # Convert to int16
    resampled_audio_int16 = (resampled_audio * 32767).astype(np.int16)
    return {"array": resampled_audio_int16, "sampling_rate": sample_rate}

dsn = "amuvarma/chatalpaca-dev"
ds = load_dataset(dsn)
ds = ds["train"].select(range(1))

# Function to process a single example
def process_example(example):
    for i in range(1, 7):  # We have ai1 to ai6
        ai_column = f'ai{i}'
        audio_column = f'ai_audio{i}'
        
        if example[ai_column] is not None:
            audio = text_to_audio(example[ai_column])
            example[audio_column] = audio
        else:
            example[audio_column] = None
    
    return example

# Add new audio columns to the dataset
for i in range(1, 7):
    ds = ds.add_column(f'ai_audio{i}', [None] * len(ds))


# Apply the processing function to the dataset with multi-processing
ds = ds.map(
    process_example,
    num_proc=10,  # Number of processes to use
    batch_size=10  # Process 10 examples at a time
)
# Cast each new column to Audio individually
for i in range(1, 7):
    ds = ds.cast_column(f'ai_audio{i}', Audio())


print(ds)

ds.push_to_hub("amuvarma/chatalpaca-dev-audio", use_temp_dir=True)
