from datasets import load_dataset
from transformers import AutoTokenizer
import datasets
import os
from datasets import DatasetDict
import string



tkn = "meta-llama/Llama-3.2-3B-Instruct"
tokeniser = AutoTokenizer.from_pretrained(tkn)

push_name = "amuvarma/luna-day3-classification"

ds_name = "amuvarma/1339-emo-class"
ds = load_dataset(ds_name)
 

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

fac_order = ['facodec_1', "facodec_0", "facodec_2",  "facodec_3",  "facodec_4",  "facodec_5"]


def process_dataset(dataset):
    # Reorder facodec columns
    
    # Function to add values to facodec columns
    def add_values(example):
        for i, col in enumerate(fac_order):
            example[col] = [x + audio_tokens_start + i * 1024 for x in example[col]]
        return example
    
    # Apply the transformations
    dataset = dataset.map(lambda x: {col: x[col] for col in fac_order})
    dataset = dataset.map(add_values)
    
    return dataset


print(ds)
ds_1 = process_dataset(ds['train'])
ds_1_test = process_dataset(ds['test'])


ds_2 = ds_1
ds_2_test = ds_1_test

def create_input_ids(example):
    input_ids =  [start_of_ai, start_of_speech]
    
    # Interleave the facodec lists
    max_len = max(len(example[facodec]) for facodec in fac_order)
    
    for i in range(max_len):
        for facodec in fac_order:
            if i < len(example[facodec]):
                input_ids.append(example[facodec][i])
    
    input_ids += [end_of_speech, end_of_ai]
    
    example['input_ids'] = input_ids
    return example


ds_3 = ds_2.map(create_input_ids)
ds_3_test = ds_2_test.map(create_input_ids)

max_length = 8192
def pad_and_create_mask(example):
    if len(example['input_ids']) > max_length:
        example['input_ids'] = example['input_ids'][:max_length]
        example['attention_mask'] = [1] * max_length
    else:
        padding_length = max_length - len(example['input_ids'])
        example['attention_mask'] = [1] * len(example['input_ids']) + [0] * padding_length
        example['input_ids'] = example['input_ids'] + [pad_token] * padding_length

    return example
ds_4 = ds_3.map(pad_and_create_mask)
ds_4_test = ds_3_test.map(pad_and_create_mask)
emotion_to_label = {
    "whisper":0, "angry":1, "sad":2, "slow":3, "curious":4, "happy":5, "surprise":6, "crying":7
}

print(ds_4_test)


def add_label_column(example):
    example['labels'] = emotion_to_label.get(example['emotion'].lower())  # Use -1 as default for unknown emotions
    return example

ds_5 = ds_4.map(
    add_label_column,
)

ds_5_test = ds_4_test.map(
    add_label_column,
)

max_length = 3072
def pad_sequence(example):
    if len(example['input_ids']) > max_length:
        example['input_ids'] = example['input_ids'][:max_length]
    else:
        padding_length = max_length - len(example['input_ids'])
        example['input_ids'] = example['input_ids'] + [pad_token] * padding_length
    return example

ds_5 = ds_5.map(pad_sequence)
ds_5_test = ds_5_test.map(pad_sequence)

columns_to_keep = ["input_ids", "attention_mask", "labels"]
all_columns = ds_5.column_names
columns_to_remove = [col for col in all_columns if col not in columns_to_keep]
dataset_to_upload = ds_5.remove_columns(columns_to_remove)
dataset_to_upload_test = ds_5_test.remove_columns(columns_to_remove)

combined_ds = DatasetDict({'train': dataset_to_upload, 'test': dataset_to_upload_test})   
combined_ds.push_to_hub(push_name)