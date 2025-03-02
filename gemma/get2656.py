from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# dsn = "amuvarma/wikipedia-unfiltered-en-tokenised-grouped-2048"
# dsn1 = "amuvarma/text-messages-6m-iids-grouped-2-2048"

dsn2 = "amuvarma/facebook-natural-reasoning-TTT"
snapshot_download(
    repo_id=dsn2,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

ds2 = load_dataset(dsn2, split='train')
print(ds2)

# ds = load_dataset(dsn, split='train')
# ds1 = load_dataset(dsn1, split='train')

# ds_full = concatenate_datasets([ds, ds1])

# ds_full = ds_full.shuffle(seed=42).shuffle(42)

# ds_full.push_to_hub(f"amuvarma/all-texts-2048-iids")

# print(ds)
# from datasets import Dataset
# from tqdm import tqdm

# chunk_size = 100_000  # Adjust as needed
# total_tokens = 0
# ds_length = len(ds)

# for start_idx in range(0, ds_length, chunk_size):
#     end_idx = min(start_idx + chunk_size, ds_length)
#     # Select a chunk
#     ds_chunk = ds.select(range(start_idx, end_idx))

#     # Map to compute token counts in this chunk
#     ds_chunk = ds_chunk.map(
#         lambda ex: {"token_count": len(ex["input_ids"])},
#         num_proc=60,  # Parallel processing
#         batched=False
#     )

#     # Sum the token_count column for this chunk
#     total_tokens += sum(ds_chunk["token_count"])

# print("Total number of tokens:", total_tokens)

# total_rows_1024 = total_tokens//1024
# total_rows_2656 = total_tokens//2656
# total_rows_2048 = total_tokens//2048

# print("Total number of rows (2656):", total_rows_2656)
# print("Total number of rows (2048):", total_rows_2048)
# print("Total number of rows (1024):", total_rows_1024)
