from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoTokenizer

tokeniser = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(tokeniser)
dsn = "wikimedia/wikipedia"
snapshot_download(
    repo_id=dsn,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,   
    allow_patterns=["20231101.en"],       
  
)
 
ds = load_dataset(dsn, "20231101.en", split="train")
ds = ds.shuffle(seed=42).shuffle(42)
ds.remove_columns(["url", "id", "title"])
# filtered_ds = ds.filter(
#     lambda x: len(x["text"]) < 35000 and len(x["text"]) > 1000,
#     num_proc=60  # Adjust based on your CPU cores
# ) 

def tokenise(example):
    example["input_ids"] = tokenizer.encode(example["text"], add_special_tokens=True)
    return example

filtered_ds = ds.map(tokenise, num_proc=60, remove_columns=ds.column_names)


filtered_ds.push_to_hub(f"amuvarma/wikipedia-unfiltered-en-tokenised")