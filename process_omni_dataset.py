from datasets import load_dataset
def _load_dataset(self, dataset_name):
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",   
        revision="main",        
        max_workers=64         
    )
    return load_dataset(dataset_name, split="train")

dsn= "gpt-omni/VoiceAssistant-400K"


myds = _load_dataset(dsn)

