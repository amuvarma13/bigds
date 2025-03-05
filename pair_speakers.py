from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # use standard tqdm
import time  # optional, for simulating processing delay

# Define the dataset repository
dsn = "amuvarma/emilia-snac-merged-with-speaker-all"

# Download the dataset snapshot from the Hugging Face Hub
snapshot_download(
    repo_id=dsn,
    repo_type="dataset",
    revision="main",
    max_workers=64,
)

# Load and sort the dataset by speaker
ds = load_dataset(dsn, split="train")
sorted_ds = ds.sort("speaker")

def pair_in_chunk(dataset, start, end, chunk_index):
    """
    Processes a chunk of the dataset from index start to end.
    Displays an inner progress bar for this chunk.
    """
    used_speakers = set()
    paired_rows = []
    i = start
    # Create a progress bar for this chunk.
    pbar = tqdm(total=(end - start), desc=f"Chunk {chunk_index}", leave=True)
    while i < end - 1:
        row1 = dataset[i]
        row2 = dataset[i + 1]
        if row1["speaker"] == row2["speaker"] and row1["speaker"] not in used_speakers:
            paired_rows.append({
                "speaker": row1["speaker"],
                "codes_list_1": row1["codes_list"],
                "codes_list_2": row2["codes_list"],
                "text_1": row1["text"],
                "text_2": row2["text"]
            })
            used_speakers.add(row1["speaker"])
            i += 2
            pbar.update(2)
        else:
            i += 1
            pbar.update(1)
        # Optionally simulate processing delay
        # time.sleep(0.001)
    pbar.close()
    return paired_rows

# Determine total number of rows and prepare chunk parameters
total = len(sorted_ds)
num_workers = 8
chunk_size = total // num_workers

# Create 8 chunks with a one-row overlap between chunks (except for the last chunk)
chunks = []
for i in range(num_workers):
    start = i * chunk_size
    end = (i + 1) * chunk_size + (1 if i < num_workers - 1 else 0)
    chunks.append((start, min(end, total)))

# Process chunks in parallel using ThreadPoolExecutor and display overall progress
paired_results = []
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [
        executor.submit(pair_in_chunk, sorted_ds, start, end, i)
        for i, (start, end) in enumerate(chunks)
    ]
    for future in tqdm(futures, desc="Processing chunks overall", unit="chunk"):
        paired_results.extend(future.result())

# Remove duplicate pairs from overlapping regions (keeping only the first pair per speaker)
final_paired = {}
for pair in paired_results:
    if pair["speaker"] not in final_paired:
        final_paired[pair["speaker"]] = pair

# Convert the final paired results into a list
paired_rows = list(final_paired.values())

# Create a new dataset from the paired rows and push it to the Hugging Face Hub
paired_dataset = Dataset.from_dict({k: [d[k] for d in paired_rows] for k in paired_rows[0]})
paired_dataset.push_to_hub("amuvarma/emilia-snac-merged-with-speaker-all-pairs")
