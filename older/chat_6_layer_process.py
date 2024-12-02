from datasets import load_dataset
from transformers import AutoTokenizer
import datasets
import os
import string
import random
 



tkn = "meta-llama/Llama-3.2-3B-Instruct"
tokeniser = AutoTokenizer.from_pretrained(tkn)
custom_tokeniser = AutoTokenizer.from_pretrained("amuvarma/llama-2.3m-full")

push_name = "amuvarma/luna-4days-instruct"

ds_name = "amuvarma/luna-4days-combined-clean-wcaps"
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

with open('read_phrases.txt', 'r') as file:
    lines = file.readlines()
    # Remove newline characters
    spoken_phrases = [line.strip() for line in lines]


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


def create_messages(example):
    instruction_phrase = random.choice(spoken_phrases)



    user_string = example["transcript"]
    user_emotion = example["emotion"]
    user_text = example["text"]


    user_text = user_text.translate(str.maketrans("", "", string.punctuation)).lower()
    instruction_phrase_string = instruction_phrase.replace("<phrase>", user_text).replace("<emotion>", user_emotion)





    input_ids = [start_of_speech]
    
    # Interleave the facodec lists
    max_len = max(len(example[facodec]) for facodec in fac_order)
    
    for i in range(max_len):
        for facodec in fac_order:
            if i < len(example[facodec]):
                input_ids.append(example[facodec][i])
    
    input_ids += [end_of_speech]

    audio_string  = custom_tokeniser.decode(input_ids, add_special_tokens=False)
    
    example['user_message'] = instruction_phrase_string
    example['assistant_message'] = user_string + audio_string
    return example


ds_2 = ds_1.map(create_messages)




columns_to_keep = ["user_message", "assistant_message"]
all_columns = ds_2.column_names

ds_2.push_to_hub(push_name)