from datasets import load_dataset
from datasets import Audio, Features, Value
from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amphion/Emilia-Dataset"
path = "Emilia/EN/*.tar"
dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")

subdataset = dataset.select(range(40000))

import json

def transform_dataset(example):

    # Parse the JSON field if it's a string
    if isinstance(example["json"], str):
        json_data = json.loads(example["json"])
    else:
        json_data = example["json"]
    
    # Extract the text from the JSON data
    text = json_data["text"]
    
    # Return a new structure with just text and audio
    return {
        "speaker": json_data["speaker"],
        "text": text,
        "audio": {"array":example["mp3"]["array"], 
                  "sampling_rate": 24000}  # Keep the original audio structure
    }



# Define the new features schema, casting 'audio' as an Audio feature.
features = Features({
    "text": Value("string"),
    "audio": Audio(sampling_rate=24000)
})
subdataset = subdataset.shuffle(seed=42)
# Apply the map with the features argument.
transformed_dataset = subdataset.map(
    transform_dataset,
    remove_columns=subdataset.column_names,
    num_proc=32,
    features=features
)

transformed_dataset = transformed_dataset.push_to_hub("amuvarma/Emilia-Dataset-0-40")
