from datasets import load_dataset

from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amphion/Emilia-Dataset"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",   
    revision="main",        
    max_workers=64, 
    allow_patterns=["Emilia/EN/*.tar"],       
)

path = "Emilia/EN/*.tar"
dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")
print(dataset[0]["json"])

# Define a mapping function to extract the speaker ID
# def extract_speaker(example):
#     # Assumes the speaker ID is located at example["json"]["speaker"]
#     return {"speaker": example["json"]["speaker"]}

# # Apply the mapping function to the dataset
# speaker_dataset = dataset.map(extract_speaker, num_proc=64, remove_columns=dataset.column_names)

# # Create a set of unique speaker IDs from the new "speaker" column
# unique_speakers = set(speaker_dataset["speaker"])

# # Print the number of unique speaker IDs
# print("Number of unique speakers:", len(unique_speakers))


dataset = dataset.select(range(1000))



from datasets import Dataset, Audio
from tqdm import tqdm  # optional, for progress indication

def pair_generator(dataset):
    # Dictionary to store one unmatched sample per speaker
    unmatched = {}
    # Iterate over the dataset; if possible, use streaming mode to avoid loading all data
    for row in tqdm(dataset, total=len(dataset)):
        speaker = row["json"]["speaker"]
        if speaker in unmatched:
            # We've seen this speaker before: yield a paired example and remove the stored sample.
            prev_row = unmatched.pop(speaker)
            yield {
                "audio_1": prev_row["mp3"],
                "text_1": prev_row["json"]["text"],
                "audio_2": row["mp3"],
                "text_2": row["json"]["text"]
            }
        else:
            # Store the row for later pairing.
            unmatched[speaker] = row

# Create the new dataset from the generator.
paired_dataset = Dataset.from_generator(lambda: pair_generator(dataset))

# Cast the audio columns to the Audio feature.
paired_dataset = paired_dataset.cast_column("audio_1", Audio())
paired_dataset = paired_dataset.cast_column("audio_2", Audio())

print(paired_dataset)
