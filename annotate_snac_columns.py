dsn = "amuvarma/va-0-10k-snac"

from datasets import load_dataset
import os
from transformers import AutoTokenizer
ds = load_dataset(dsn, split='train')



tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai =  tokeniser_length + 6
pad_token = tokeniser_length + 7

audio_tokens_start = tokeniser_length + 10


# Remove all columns except "answer_snac"
# columns_to_remove = [col for col in ds.column_names if col != "answer_snac"]
# ds = ds.remove_columns(columns_to_remove)

def convert_string_to_codes(example):
    snac_str = example["answer_snac"]
    parts = snac_str.split('#')
    parts = [p.strip() for p in parts if p.strip()]
    snac_lols = [list(map(int, p.split())) for p in parts]
    example["snac_lols"] = snac_lols
    return example

# Determine number of processes based on CPU count
num_proc = os.cpu_count()

# Map the function in parallel
ds = ds.map(convert_string_to_codes, num_proc=num_proc)

print(ds)