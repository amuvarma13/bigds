from datasets import load_dataset
from transformers import AutoTokenizer
import datasets
import os


tkn = "meta-llama/Llama-3.2-3B-Instruct"
tokeniser = AutoTokenizer.from_pretrained(tkn)

push_name = "amuvarma/370k-1"

ds_name = "amuvarma/stt-370k-raw-1dups"
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

ds_1 = process_dataset(ds['train'])

def tokenize_and_add_to_dataset(dataset):
    def tokenize_transcript(example):
        # Tokenize the transcript
        tokenized = tokeniser(example['transcript'])
        

        tokenized_text = tokenized['input_ids'] + [end_of_text]
        
        # Add the new tokenised_text to the example
        example['tokenised_text'] = tokenized_text
        
        return example

    # Apply the tokenization to the dataset
    tokenized_dataset = dataset.map(tokenize_transcript)
    
    return tokenized_dataset

ds_2 = tokenize_and_add_to_dataset(ds_1)

def create_input_ids(example):
    input_ids = [start_of_human, start_of_speech]
    
    # Interleave the facodec lists
    max_len = max(len(example[facodec]) for facodec in fac_order)
    
    for i in range(max_len):
        for facodec in fac_order:
            if i < len(example[facodec]):
                input_ids.append(example[facodec][i])
    
    input_ids += [end_of_speech,end_of_human, start_of_ai]

    input_ids += example['tokenised_text'] + [end_of_human]
    
    example['input_ids'] = input_ids
    return example


ds_3 = ds_2.map(create_input_ids)

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

def preprocess_function(examples):
    examples['labels'] = [
        [(token_id if token_id != pad_token else -100) for token_id in input_ids]
        for input_ids in examples['input_ids']
    ]
    return examples


# Determine the number of CPU cores available
num_cpus = os.cpu_count()

# Use 75% of available CPUs, but at least 1
num_processes = max(1, int(num_cpus * 0.75))

ds_5 = ds_4.map(
    preprocess_function,
    batched=True,
    num_proc=num_processes,
    desc="Preprocessing dataset"
)

columns_to_keep = ["input_ids", "attention_mask", "labels"]
all_columns = ds_5.column_names
columns_to_remove = [col for col in all_columns if col not in columns_to_keep]
dataset_to_upload = ds_5.remove_columns(columns_to_remove)



dataset_to_upload.push_to_hub(push_name)