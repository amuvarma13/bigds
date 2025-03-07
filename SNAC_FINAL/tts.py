dsn = "amuvarma/enhanced_combined_with_source"

from datasets import load_dataset
import os
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=dsn,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

ds = load_dataset(dsn, split='train')


push_name = f"{dsn}-TTS-src"

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

tokenizer_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


num_proc = os.cpu_count() - 2


#filter out all rows without codes list  
ds = ds.filter(lambda x: x["codes_list"] is not None)
ds = ds.filter(lambda x: len(x["codes_list"]) > 0)



def create_input_ids(example):
    text_ids = tokenizer.encode(example["text"] + " " + f'<{example["row"]}>', add_special_tokens=True)
    text_ids.append(end_of_text)
    example["text_tokens"] = text_ids
    input_ids = (
        [start_of_human]
        + example["text_tokens"]
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
    )
    example["input_ids"] = input_ids

    return example

ds = ds.map(create_input_ids, num_proc=20, remove_columns=["text", "codes_list"])

columns_to_keep = ["input_ids"]
columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]

ds = ds.remove_columns(columns_to_remove)
print(ds.column_names)

ds = ds.shuffle(seed=42).shuffle(seed=42)


ds.push_to_hub(push_name)