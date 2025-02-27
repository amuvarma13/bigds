from datasets import load_dataset

from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amphion/Emilia-Dataset"
path = "Emilia/EN/*.tar"
dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")

dataset = dataset.select(range(10))



print(dataset[0]["json"])
from datasets import Dataset, Audio
from tqdm import tqdm

# Initialize the Audio feature (you can set sampling_rate if desired)
audio_feature = Audio()

def decode_audio(file_val):
    """
    Ensure that file_val is passed as a dictionary with both "path" and "bytes" keys.
    If file_val is a string, wrap it accordingly.
    """
    if isinstance(file_val, str):
        # Wrap the string file path into a dictionary format
        file_dict = {"path": file_val, "bytes": None}
    elif isinstance(file_val, dict):
        # Ensure the "bytes" key exists in the dictionary
        file_dict = file_val.copy()
        if "bytes" not in file_dict:
            file_dict["bytes"] = None
    else:
        raise ValueError("Unsupported type for audio file")
    return audio_feature.decode_example(file_dict)

def pair_generator(dataset):
    unmatched = {}
    # Iterate over the dataset in a memory-efficient manner.
    for row in tqdm(dataset, total=len(dataset)):
        speaker = row["json"]["speaker"]
        if speaker in unmatched:
            prev_row = unmatched.pop(speaker)
            yield {
                "audio_1": decode_audio(prev_row["mp3"]),
                "text_1": prev_row["json"]["text"],
                "audio_2": decode_audio(row["mp3"]),
                "text_2": row["json"]["text"]
            }
        else:
            unmatched[speaker] = row

# Create the new dataset from the generator.
paired_dataset = Dataset.from_generator(lambda: pair_generator(dataset))

print(paired_dataset)
