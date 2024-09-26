from datasets import load_dataset
from transformers import AutoTokenizer
import datasets


tok_name = "google/gemma-2-2b"
tokeniser = AutoTokenizer.from_pretrained(tok_name)

push_name = "amuvarma/6_layer_interleave-102345-500k-0"

ds_name = "amuvarma/raw_500k_0"
ds = load_dataset(ds_name)

start_of_human = 256001
end_of_human = 256002
start_of_ai =256003
end_of_ai = 256004
start_of_speech = 256005
end_of_speech = 256006


def process_dataset(dataset):
    # Reorder facodec columns
    new_order = ['facodec_1', 'facodec_0', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5']
    
    # Function to add values to facodec columns
    def add_values(example):
        for i, col in enumerate(new_order):
            example[col] = [x + 256010 + i * 1024 for x in example[col]]
        return example
    
    # Apply the transformations
    dataset = dataset.map(lambda x: {col: x[col] for col in new_order})
    dataset = dataset.map(add_values)
    
    return dataset

ds_1 = process_dataset(ds['train'])

def tokenize_and_add_to_dataset(dataset):
    def tokenize_transcript(example):
        # Tokenize the transcript
        tokenized = tokeniser(example['transcript'])
        
        # Append token 1 to the tokenized text
        tokenized_text = tokenized['input_ids'] + [1]
        
        # Add the new tokenised_text to the example
        example['tokenised_text'] = tokenized_text
        
        return example

    # Apply the tokenization to the dataset
    tokenized_dataset = dataset.map(tokenize_transcript)
    
    return tokenized_dataset

ds_2 = tokenize_and_add_to_dataset(ds_1)

def create_input_ids(example):
    input_ids = [start_of_human] + example['tokenised_text'] + [end_of_human, start_of_ai]
    
    # Interleave the facodec lists
    facodec_order = ['facodec_1', 'facodec_0', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5']
    max_len = max(len(example[facodec]) for facodec in facodec_order)
    
    for i in range(max_len):
        for facodec in facodec_order:
            if i < len(example[facodec]):
                input_ids.append(example[facodec][i])
    
    input_ids += [end_of_speech, end_of_ai]
    
    example['input_ids'] = input_ids
    return example


ds_3 = ds_2.map(create_input_ids)

max_length = 8192
def pad_and_create_mask(example):
    # Assume max_length is defined globally in the notebook
    
    # Pad or truncate input_ids
    if len(example['input_ids']) > max_length:
        example['input_ids'] = example['input_ids'][:max_length]
    else:
        padding_length = max_length - len(example['input_ids'])
        example['input_ids'] = example['input_ids'] + [0] * padding_length
    
    # Create attention_mask
    example['attention_mask'] = [1] * len(example['input_ids']) + [0] * (max_length - len(example['input_ids']))
    
    # Ensure attention_mask is also of length max_length
    example['attention_mask'] = example['attention_mask'][:max_length]
    
    return example
ds_4 = ds_3.map(pad_and_create_mask)


ds_4.push_to_hub(push_name)