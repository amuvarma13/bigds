from datasets import load_dataset
from transformers import AutoTokenizer
import datasets
import os
import string
import random



tkn = "meta-llama/Llama-3.2-3B-Instruct"
tokeniser = AutoTokenizer.from_pretrained(tkn)

push_name = "amuvarma/26k-stts-duplex-convos-raw-fac-1dups-proc-train-col-clean"

ds_name = "amuvarma/26k-stts-duplex-convos-raw-fac-1dups-contentonly"
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

fac_order = ['facodec_1',  'facodec_0', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5']


def read_instructions(filename):
    instructions = []
    with open(filename, 'r') as file:
        instructions = file.read().splitlines()
    return instructions

instruction_list = read_instructions('read_out.txt')

def process_dataset(dataset):

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

        random_instruction = random.choice(instruction_list)
        tokenized = tokeniser(
           random_instruction+ " "+example["answer"])
        

        tokenized_text = tokenized['input_ids'] + [end_of_text]
        
        # Add the new tokenised_text to the examsple
        example['tokenised_text'] = tokenized_text
        
        return example

    # Apply the tokenization to the dataset
    tokenized_dataset = dataset.map(tokenize_transcript)
    
    return tokenized_dataset

ds_2 = tokenize_and_add_to_dataset(ds_1)

def create_input_ids(example):
    input_ids = [start_of_human] + example['tokenised_text'] + [end_of_human, start_of_ai, start_of_speech]
    
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





# num_cpus = os.cpu_count()

# num_processes = max(1, int(num_cpus * 0.75))

# ds_5 = ds_4.map(
#     preprocess_function,
#     batched=True,
#     num_proc=num_processes,
#     desc="Preprocessing dataset"
# )

columns_to_keep = ["input_ids"]
all_columns = ds_3.column_names
columns_to_remove = [col for col in all_columns if col not in columns_to_keep]
dataset_to_upload = ds_3.remove_columns(columns_to_remove)

dataset_to_upload.select(range(0,1000000))

dataset_to_upload.push_to_hub(push_name)