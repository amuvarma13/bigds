from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

dsn = "amuvarma/text-messages-6m"

snapshot_download(
    repo_id=dsn,
    repo_type="dataset",   
    revision="main",        
    max_workers=64,     
)

ds = load_dataset(dsn, split='train')
ds = ds.select(range(10000))

tokeniser_length = 256000
start_of_text = 2
end_of_text = tokeniser_length + 8

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai =  tokeniser_length + 6
pad_token = tokeniser_length + 7

audio_tokens_start = tokeniser_length + 10

tokenizer_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def create_input_ids(example):
    row_ids = []
    for m in example["messages"]:
        content = m["content"]
        role = m["role"]

        content_ids = tokenizer.encode(content, add_special_tokens=True)
        content_ids.append(end_of_text)

        if role == "user":
            row_ids.extend(start_of_human)
            row_ids += content_ids
            row_ids.extend(end_of_human)
        else:
            row_ids.extend(start_of_ai)
            row_ids += content_ids
            row_ids.extend(end_of_ai)

    example["input_ids"] = row_ids
    return example

ds = ds.map(create_input_ids, num_proc=20, remove_columns=ds.column_names)

ds.push_to_hub(f"amuvarma/{dsn}-iids")