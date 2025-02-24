## TAKES IN DATASET WITH COLUMNS codes_list, question, answer

dsn = "amuvarma/luna-all-raw-snacced"

from datasets import load_dataset
import os
from transformers import AutoTokenizer
ds = load_dataset(dsn, split='train')


push_name = "amuvarma/amuvarma/luna-all-raw-snacced-TTS"

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

# ds = ds.filter(lambda x: x['question'] and x['answer'] and x['codes_list'])

#filter out all rows with codeslist length over 12000
ds = ds.filter(lambda x: len(x['codes_list']) < 8192)

# Map the function in parallel
def tokenize_fn(example):
    text_ids = tokenizer.encode(example["text"], add_special_tokens=True)
    text_ids.append(end_of_text)
    example["text_tokens"] = text_ids
    return example

ds = ds.map(tokenize_fn, num_proc=num_proc)

def create_input_ids(example):
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
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)
    return example

ds = ds.map(create_input_ids, num_proc=num_proc)

columns_to_keep = ["input_ids", "labels",   "attention_mask"]
columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]

ds = ds.remove_columns(columns_to_remove)
print(ds.column_names)


ds.push_to_hub(push_name)