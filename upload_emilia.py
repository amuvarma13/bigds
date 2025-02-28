from datasets import load_dataset
from datasets import Audio, Features, Value
from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_id = "amphion/Emilia-Dataset"
path = "Emilia/EN/*.tar"
dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")

subdataset = dataset.select(range(1000))
subdataset.push_to_hub("amuvarma/Emilia-Dataset-1000")
 
# import json

# def transform_dataset(example):
#     try:
#         # Safely parse the JSON field
#         json_field = example["json"]
#         if isinstance(json_field, str):
#             json_data = json.loads(json_field)
#         else:
#             json_data = json_field

#         # Safely extract expected keys
#         text = json_data.get("text", "")
#         speaker = json_data.get("speaker", "unknown")

#         return {
#             "speaker": speaker,
#             "text": text,
#             "audio": {
#                 "array": example["mp3"]["array"],
#                 "sampling_rate": 24000
#             }
#         }
#     except Exception as e:
#         # Log the error and optionally return a default value or skip this example
#         print(f"Skipping example due to error: {e}")
#         return None  # Returning None will drop this example from the output.



# # Define the new features schema, casting 'audio' as an Audio feature.
# features = Features({
#     "text": Value("string"),
#     "audio": Audio(sampling_rate=24000), 
#     "speaker": Value("string")
# })
# subdataset = subdataset.shuffle(seed=42)
# # Apply the map with the features argument.
# transformed_dataset = subdataset.map(
#     transform_dataset,
#     remove_columns=subdataset.column_names,
#     num_proc=84,
#     features=features
# )

# transformed_dataset = transformed_dataset.push_to_hub("amuvarma/Emilia-Dataset-0-100")
