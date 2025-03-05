from datasets import load_dataset, Dataset
from tqdm import tqdm
from huggingface_hub import snapshot_download

dsn = "amuvarma/emilia-snac-merged-with-speaker-all"

snapshot_download(
        repo_id=dsn,
        repo_type="dataset",   
        revision="main",        
        max_workers=64,     
)

ds = load_dataset(dsn, split="train")

sorted_ds = ds.sort("speaker")


def pair_consecutive_rows(dataset):
    used_speakers = set()   # Ensure each speaker is used only once
    paired_rows = []        # Store the new paired rows
    i = 0
    total = len(dataset)

    with tqdm(total=total, desc="Pairing rows", unit="row") as pbar:
        while i < total - 1:
            row1 = dataset[i]
            row2 = dataset[i + 1]

            # Check if consecutive rows have the same speaker and haven't been used yet
            if row1["speaker"] == row2["speaker"] and row1["speaker"] not in used_speakers:
                paired_rows.append({
                    "speaker": row1["speaker"],
                    "codes_list_1": row1["codes_list"],
                    "codes_list_2": row2["codes_list"],
                    "text_1": row1["text"],
                    "text_2": row2["text"]
                })
                used_speakers.add(row1["speaker"])
                i += 2  # Skip the next row since it has been paired
                pbar.update(2)
            else:
                i += 1
                pbar.update(1) 

    return paired_rows

# Example usage:
# Assume `ds` is your loaded Hugging Face dataset converted to a list of dicts.
paired_rows = pair_consecutive_rows(sorted_ds)

# Create a new dataset from the paired rows
paired_dataset = Dataset.from_dict({k: [d[k] for d in paired_rows] for k in paired_rows[0]})


paired_dataset.push_to_hub("amuvarma/emilia-snac-merged-with-speaker-all-pairs")