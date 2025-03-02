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
from datasets import Dataset
from tqdm import tqdm

chunk_size = 100_000  # Adjust as needed
total_tokens = 0
ds_length = len(ds)

for start_idx in range(0, ds_length, chunk_size):
    end_idx = min(start_idx + chunk_size, ds_length)
    # Select a chunk
    ds_chunk = ds.select(range(start_idx, end_idx))

    # Map to compute token counts in this chunk
    ds_chunk = ds_chunk.map(
        lambda ex: {"token_count": len(ex["input_ids"])},
        num_proc=60,  # Parallel processing
        batched=False
    )

    # Sum the token_count column for this chunk
    total_tokens += sum(ds_chunk["token_count"])

print("Total number of tokens:", total_tokens)
