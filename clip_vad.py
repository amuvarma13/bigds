from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import os
model = load_silero_vad()

dsn = "amuvarma/luna-48k-full-enhanced_emotion_mapped"
from datasets import load_dataset
ds = load_dataset(dsn, split="train")

import torch
import torchaudio
from torchaudio.transforms import Resample

from datasets import load_dataset, Audio, Features, Value

#add casting


resampler = torchaudio.transforms.Resample(48000, 16000)

def process_row(row):
    # Get the enhanced audio array
    wav = row["audio"]["array"]

    # Explicitly convert to the right dtype before resampling
    wav_tensor = torch.from_numpy(wav).to(torch.float32)

    # Resample for Silero VAD model
    silero_wav = resampler(wav_tensor).squeeze().to("cuda")

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        silero_wav,
        model,
        return_seconds=True,
    )

    # Crop audio to speech section with small padding
    start_time = max(0, speech_timestamps[0]["start"] - 0.1)
    end_time = speech_timestamps[-1]["end"] + 0.1
    cropped_speech = wav[int(start_time * 48000):int(end_time * 48000)]

    # Build a new dictionary with only the keys specified in Features
    return {
        "text": row["text"],
        "emotion": row.get("emotion", ""),  # returns empty string if not present
        "audio": {
            "array": cropped_speech,
            "sampling_rate": 48000
        }
    }


# Apply the processing with proper casting
audio_feature = Audio(sampling_rate=48000)
new_features = Features({
    "audio": audio_feature,
    "text": Value("string"),
    "emotion": Value("string")
})

processed_dataset = ds.map(
    process_row,
    features=new_features,
    remove_columns=ds.column_names,
    num_proc=64,
)

processed_dataset.push_to_hub("amuvarma/luna-48k-full-enhanced_emotion_mapped-clipped")


