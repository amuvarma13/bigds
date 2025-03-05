from datasets import load_dataset, Dataset
from tqdm import tqdm
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor

# Define the dataset repository
dsn = "amuvarma/emilia-snac-merged-with-speaker-all"

# Download the dataset snapshot from the Hugging Face Hub
snapshot_download(
    repo_id=dsn,
    repo_type="dataset",
    revision="main",
    max_workers=64,
)

# Load the dataset and sort by speaker
ds = load_dataset(dsn, split="train")
sorted_ds = ds.sort("speaker")

# Define a helper function to pair consecutive rows within a chunk
def pair_in_chunk(dataset, start, end):
    used_speakers = set()   # Ensure each speaker is used only once in this chunk
    paired_rows = []        # Store the new paired rows
    i = start
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
            i += 2  # Skip the paired row
        else:
            i += 1
    return paired_rows

# Determine total number of rows and chunk parameters
total = len(sorted_ds)
num_workers = 8
chunk_size = total // num_workers

# Create 8 chunks with one-row overlap between chunks (except the last chunk)
chunks = []
for i in range(num_workers):
    start = i * chunk_size
    end = (i + 1) * chunk_size + (1 if i < num_workers - 1 else 0)
    chunks.append((start, min(end, total)))

# Process the chunks in parallel using ThreadPoolExecutor
paired_results = []
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(pair_in_chunk, sorted_ds, start, end) for start, end in chunks]
    for future in tqdm(futures, desc="Processing chunks"):
        paired_results.extend(future.result())

# Remove duplicates from overlapping regions by keeping only the first pair for each speaker
final_paired = {}
for pair in paired_results:
    if pair["speaker"] not in final_paired:
        final_paired[pair["speaker"]] = pair

# Convert the final paired results into a list
paired_rows = list(final_paired.values())

# Create a new dataset from the paired rows
paired_dataset = Dataset.from_dict({k: [d[k] for d in paired_rows] for k in paired_rows[0]})

# Push the new paired dataset to the Hugging Face Hub
paired_dataset.push_to_hub("amuvarma/emilia-snac-merged-with-speaker-all-pairs")
