from datasets import load_dataset
import random
from transformers import AutoTokenizer
import numpy as np
from datasets import concatenate_datasets

dataset_name = "amuvarma/1m-fac_0"
split = 'train'

dataset = load_dataset(dataset_name, split="train")
editable_dataset = dataset

#first lets remove duplicates from the dataset
from itertools import groupby
 
def remove_consecutive_duplicates(tokens, n=3):
    result = []
    for k, g in groupby(tokens):
        group = list(g)
        if len(group) < n:
            result.extend(group)
        else:
            result.append(k)
    return result

def process_batch(examples):
    examples['audio_tokens'] = [remove_consecutive_duplicates(tokens) for tokens in examples['facodec_1']]
    return examples

editable_dataset = editable_dataset.map(
    process_batch,
    batched=True,
    batch_size=1000,  # Adjust this based on your memory constraints
    num_proc=4  # Adjust based on your CPU cores
)
#now lets format the input_ids into speech segments which involve adding start of speech and end of speech tokens
def add_to_tokens_batch(examples):
    examples['audio_tokens'] = [
        [token + 256003 for token in tokens]
        for tokens in examples['audio_tokens']
    ]
    return examples

editable_dataset = editable_dataset.map(
    add_to_tokens_batch,
    batched=True,
    batch_size=1000,  # You can adjust this value
    num_proc=4  # Adjust based on your CPU cores
)

#now lets define some token_ids
start_of_text = 2
end_of_text = 1

start_of_speech = 256001
end_of_speech = 256002

start_of_human = 256000 + 1024 + 4 + 1
end_of_human = 256000 + 1024 + 4 + 2

start_of_ai = 256000 + 1024 + 4 + 3
end_of_ai = 256000 + 1024 + 4 + 4

num_samples = len(editable_dataset)
midpoint = num_samples // 2
first_half = editable_dataset.select(range(midpoint))

with open('./tts.txt', 'r', encoding='utf-8') as file:
    tts_list = file.read().splitlines()



def add_instructions(examples):
    num_examples = len(examples['transcript'])

    # Randomly choose instructions for each example
    random_instructions = random.choices(tts_list, k=num_examples)

    # Combine instructions with transcripts
    examples['instructions'] = [
        f"{instruction}\n{transcript}"
        for instruction, transcript in zip(random_instructions, examples['transcript'])
    ]

    return examples

# Apply the function to the dataset
first_half_with_instructions = first_half.map(
    add_instructions,
    batched=True,
    batch_size=1000,  # Adjust based on your memory constraints
    num_proc=4  # Adjust based on your CPU cores
)



tokenizer_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def process_example(example):
    tokenized = tokenizer(example['instructions'], truncation=False)

    input_ids = tokenized['input_ids'] + [1]

    input_ids = [start_of_human] + input_ids + [end_of_human]

    input_ids = input_ids + [start_of_ai] + [start_of_speech] + example['audio_tokens'] + [end_of_speech] + [end_of_ai]

    return {
        'input_ids': input_ids,
        'transcript': example['transcript'],
        'audio_tokens': example['audio_tokens']
    }


processed_first_half = first_half_with_instructions.map(
    process_example,
    remove_columns=['instructions'],
    num_proc=4
)

second_half_dataset = editable_dataset.select(range(midpoint, num_samples))


with open('./stt.txt', 'r', encoding='utf-8') as file:
    stt_list = file.read().splitlines()




def add_instructs(example):
    random_instruction = random.choice(stt_list) 

    tokenized = tokenizer(random_instruction, truncation=False)
    instructs = tokenized['input_ids'] + [1]

    return {'instructs': instructs}

# Apply the function to add the 'instructs' column
second_half_dataset = second_half_dataset.map(
    add_instructs,
    num_proc=4  # Adjust based on your CPU cores
)


def create_input_ids(example):
    transcript_tokens = tokenizer(example['transcript'], truncation=False)['input_ids'] + [1]

    if random.choice([True, False]):
        middle_sequence = [start_of_speech] + example['audio_tokens'] + [end_of_speech]  + example['instructs']
    else:
        middle_sequence = example['instructs'] + [start_of_speech] + example['audio_tokens'] + [end_of_speech]

    input_ids = (
        [start_of_human] +

        middle_sequence +
        [end_of_human] +
        [start_of_ai] +
        transcript_tokens +
        [end_of_ai]
    )

    # Return a dictionary with input_ids and the columns we want to keep
    return {
        'input_ids': input_ids,
        'transcript': example['transcript'],
        'audio_tokens': example['audio_tokens']
    }

# Apply the function to add the 'input_ids' column while keeping 'transcript' and 'audio_tokens'
second_half_dataset_processed = second_half_dataset.map(
    create_input_ids,
    remove_columns=['instructs'],  # Remove only the 'instructs' column
    num_proc=4  # Adjust based on your CPU cores
)



# First, let's add the 'type' column to each dataset
def add_type_column(example, type_value):
    return {'type': type_value}

processed_first_half_with_type = processed_first_half.map(
    lambda example: add_type_column(example, 'tts'),
    num_proc=4
)

second_half_dataset_processed_with_type = second_half_dataset_processed.map(
    lambda example: add_type_column(example, 'stt'),
    num_proc=4
)

# Now, let's combine the datasets
full_processed = concatenate_datasets([
    processed_first_half_with_type,
    second_half_dataset_processed_with_type
])

# Ensure we only keep the columns we want
full_processed = full_processed.select_columns(['transcript', 'audio_tokens', 'input_ids', 'type'])

# Shuffle the dataset to mix the 'tts' and 'stt' examples
full_processed = full_processed.shuffle(seed=42)  # Use a seed for reproducibility


def get_length(example):
    return {'length': len(example['input_ids'])}

# Add a 'length' column to the dataset
full_processed_with_length = full_processed.map(
    get_length,
    num_proc=4,  # Adjust based on your CPU cores
    batched=False
)

# Get the 'length' column as a list and find the maximum
lengths = full_processed_with_length.select_columns(['length'])['length']
max_length = max(lengths)



def pad_and_create_mask(example):
    padded_input_ids = example['input_ids'] + [0] * (max_length - len(example['input_ids']))

    # Create attention_mask
    attention_mask = [1] * len(example['input_ids']) + [0] * (max_length - len(example['input_ids']))

    return {
        'input_ids': padded_input_ids,
        'attention_mask': attention_mask
    }

# Apply padding and create attention mask
full_processed_padded = full_processed.map(
    pad_and_create_mask,
    num_proc=4  # Adjust based on your CPU cores
)

full_processed_padded.push_to_hub(
    "amuvarma/1m-crossmodal-0-dups",
)