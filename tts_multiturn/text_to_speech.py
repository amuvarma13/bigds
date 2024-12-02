import numpy as np
from openai import OpenAI
import io
from scipy.io import wavfile
import librosa
from pydub import AudioSegment

def generate_and_resample_speech(text, voice="shimmer", target_sample_rate=16000):
    client = OpenAI()
    target_sr = target_sample_rate

    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="mp3"
    )

    audio_data = response.content

    audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
    audio_np = np.array(audio.get_array_of_samples()).astype(np.float32)

    if audio.channels == 2:
        audio_np = audio_np.reshape((-1, 2)).mean(axis=1)

    audio_np = audio_np / np.max(np.abs(audio_np))

    resampled_audio = librosa.resample(audio_np, orig_sr=audio.frame_rate, target_sr=target_sr).astype(np.float32)

    return resampled_audio, target_sr