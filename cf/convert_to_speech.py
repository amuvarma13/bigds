import numpy as np
import random
import outetts
import librosa
import torch
# Configure the model
model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  # Supported languages in v0.2: en, zh, ja, ko
    dtype=torch.bfloat16,
    additional_model_config={
        'attn_implementation': "flash_attention_2"
    }
)

# Initialize the interface
interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)

# Optional: Create a speaker profile (use a 10-15 second audio clip)
add_speaker = interface.create_speaker(
    audio_path="spks.wav",
    transcript="I'm so disgusted by the state of that person's appartment."
)


default_names = ["male_1", "male_2","male_3", "female_1", "female_2", ]
all_speakers = [add_speaker]

for name in default_names:
    sp = interface.load_default_speaker(name=name)
    all_speakers.append(sp)

def convert_to_speech(prompt, speaker_index=0):


    speaker = all_speakers[speaker_index]
    output = interface.generate(
        text=prompt,
        temperature=0.1,
        repetition_penalty=1.1,
        max_length=4096,
        speaker=speaker,
    )
    audio_numpy = output.audio.squeeze().cpu().numpy()

    # Resample using librosa
    resampled_audio = librosa.resample(y=audio_numpy, orig_sr=24000, target_sr=16000)


    return resampled_audio