## TAKES IN DATASET WITH COLUMNS codes_list, question, answer

dsn = "amuvarma/sm-template-audio-snacced-2"

from datasets import load_dataset
import os
from transformers import AutoTokenizer
ds = load_dataset(dsn, split='train')


push_name = "amuvarma/sm-template-audio-snacced-TTTTMS-text"

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
start_of_motion = 128255
end_of_motion = 128254

vq_offset = 128266+(7*4096)+10

audio_tokens_start = tokeniser_length + 10

tokenizer_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


# Remove all columns except "answer_snac"
# columns_to_remove = [col for col in ds.column_names if col != "answer_snac"]
# ds = ds.remove_columns(columns_to_remove)

# Determine number of processes based on CPU count
num_proc = os.cpu_count() - 2

#filter out all rows without question, answer, or codes_list 
ds = ds.filter(lambda x: x['question'] and x['answer'] and x['codes_list'])

#filter out all rows with codeslist length over 12000
ds = ds.filter(lambda x: len(x['codes_list']) < 8192)

# Map the function in parallel
def tokenize_fn(example):
    user_ids = tokenizer.encode(example["question"], add_special_tokens=True)
    answer_ids = tokenizer.encode(example["answer"], add_special_tokens=True)
    user_ids.append(end_of_text)
    answer_ids.append(end_of_text)
    example["user_tokens"] = user_ids
    example["answer_tokens"] = answer_ids
    return example

ds = ds.map(tokenize_fn, num_proc=num_proc)

def offset_vq(example):
    example["vq_encoded_offset"] = [x+vq_offset for x in example["vq_encoded"]]
    return example

ds = ds.map(offset_vq, num_proc=num_proc)

def get_answer_labels(input_ids):
    labels = [-100] * len(input_ids)
    try:
        start_idx = input_ids.index(start_of_ai)
        end_idx = input_ids.index(start_of_motion, start_idx)
    except ValueError:
        return labels
    for i in range(start_idx, end_idx):
        labels[i] = input_ids[i]
    return labels


def create_input_ids(example):
    input_ids = (
        [start_of_human]
        + example["user_tokens"]
        + [end_of_human]
        + [start_of_ai]
        + example["answer_tokens"]
        + [start_of_motion]
        + example["vq_encoded_offset"]
        + [end_of_motion]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
    )
    example["input_ids"] = input_ids
    example["labels"] = get_answer_labels(input_ids)
    example["attention_mask"] = [1] * len(input_ids)
    return example

ds = ds.map(create_input_ids, num_proc=num_proc)

columns_to_keep = ["input_ids", "labels",   "attention_mask"]
columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]

ds = ds.remove_columns(columns_to_remove)
print(ds.column_names)


ds.push_to_hub(push_name)