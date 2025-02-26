from datasets import load_dataset
from huggingface_hub import snapshot_download
def _load_dataset(dataset_name):
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",   
        revision="main",        
        max_workers=64         
    )
    return load_dataset(dataset_name, split="train")

dsn= "gpt-omni/VoiceAssistant-400K"


myds = _load_dataset(dsn)



columns_to_keep = ["answer", "split_name"]

columns_to_remove = [col for col in myds.column_names if col not in columns_to_keep]

myds = myds.remove_columns(columns_to_remove)

#filter such that we remove any rows whose split_name is "identity"
myds = myds.filter(lambda x: x["split_name"] != "identity")


columns_to_keep = ["answer"]

columns_to_remove = [col for col in myds.column_names if col not in columns_to_keep]

myds = myds.remove_columns(columns_to_remove)


myds.push_to_hub("voice-assistant-texts-only")

print(myds)

# columns_to_keep = ["question_audio", "answer"]
# columns_to_remove = [col for col in myds.column_names if col not in columns_to_keep]
# myds = myds.remove_columns(columns_to_remove)

# myds.push_to_hub("voice-assistant-adapted")

