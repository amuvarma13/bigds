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
print(dataset)

dataset = dataset.select(range(1000))
# Define a mapping function to extract the speaker ID
def extract_speaker(example):
    # Assumes the speaker ID is located at example["json"]["speaker"]
    return {"speaker": example["json"]["speaker"]}

# Apply the mapping function to the dataset
speaker_dataset = dataset.map(extract_speaker)

# Create a set of unique speaker IDs from the new "speaker" column
unique_speakers = set(speaker_dataset["speaker"])

# Print the number of unique speaker IDs
print("Number of unique speakers:", len(unique_speakers))
