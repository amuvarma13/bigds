from datasets import load_dataset
from transformers import AutoTokenizer
import datasets
import os
import string
import random


tkn = "meta-llama/Llama-3.2-3B-Instruct"
tokeniser = AutoTokenizer.from_pretrained(tkn)

push_name = "amuvarma/dev-qa_large_0_4_speechqa-both-full-facodec-1dups-6rows-proc"

ds_name = "amuvarma/qa_large_0_4_speechqa-both-full-facodec-1dups-6rows"
ds = load_dataset(ds_name, split="train")
ds = ds.select(range(6))

tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai = tokeniser_length + 6
pad_token = tokeniser_length + 7

start_of_system = tokeniser_length + 8
end_of_system = tokeniser_length + 9


audio_tokens_start = tokeniser_length + 10


def read_instructions(filename):
    instructions = []
    with open(filename, 'r') as file:
        instructions = file.read().splitlines()
    return instructions


def process_dataset(dataset, fac_order):    # Add fac_order as parameter
    def add_values(example, fac_order):
        for i, col in enumerate(fac_order):
            if example[col]:
                example[col] = [
                    x + (i*1024) + audio_tokens_start for x in example[col]]
        return example

    # Apply the transformations
    dataset = dataset.map(lambda x: {col: x[col] for col in fac_order})
    dataset = dataset.map(lambda x: add_values(x, fac_order=fac_order))

    return dataset


# Then use it like this:
ds_1 = process_dataset(ds, ['facodec_1', 'facodec_0', 'facodec_2', 'facodec_3', 'facodec_4', 'facodec_5'])
# ds_2 = process_dataset(ds_1, ['ass2_facodec_1', 'ass2_facodec_0',
#                        'ass2_facodec_2', 'ass2_facodec_3', 'ass2_facodec_4', 'ass2_facodec_5'])
# ds_3 = process_dataset(ds_2, ['ass3_facodec_1', 'ass3_facodec_0',
#                        'ass3_facodec_2', 'ass3_facodec_3', 'ass3_facodec_4', 'ass3_facodec_5'])
# interleave the facodec columns in the order ass1_facodec_1, ass1_facodec_2... ass1_facodec_5 call the new columns ass1_facodec_interleaved
# interleave the facodec columns in the order ass2_facodec_1, ass2_facodec_2... ass2_facodec_5 call the new columns ass2_facodec_interleaved
# interleave the facodec columns in the order ass3_facodec_1, ass3_facodec_2... ass3_facodec_5 call the new columns ass3_facodec_interleaved


def interleave_facodec_columns(dataset):
    def interleave(example):
        # Interleave ass1_facodec columns in order 1 to 5
        interleaved = []
        for j in range(len(example["facodec_1"])):
            # Define the desired order for ass3_facodec columns
            for i in [1, 0, 2, 3, 4, 5]:
                # Ensure that the column exists before appending
                key = f"facodec_{i}"
                if key in example:
                    interleaved.append(example[key][j])
                else:
                    # Handle missing columns if necessary
                    interleaved.append(None)  # or some default value
        example["ass1_facodec_interleaved"] = interleaved

        # Interleave ass2_facodec columns in order 1 to 5
        # interleaved = []
        # for j in range(len(example["ass2_facodec_1"])):
        #     # Define the desired order for ass3_facodec columns
        #     for i in [1, 0, 2, 3, 4, 5]:
        #         # Ensure that the column exists before appending
        #         key = f"ass2_facodec_{i}"
        #         if key in example:
        #             interleaved.append(example[key][j])
        #         else:
        #             # Handle missing columns if necessary
        #             interleaved.append(None)  # or some default value
        # example["ass2_facodec_interleaved"] = interleaved

        # # Interleave ass3_facodec columns in the specific order 1, 0, 2, 3, 4, 5
        # interleaved = []
        # for j in range(len(example["ass3_facodec_1"])):
        #     # Define the desired order for ass3_facodec columns
        #     for i in [1, 0, 2, 3, 4, 5]:
        #         # Ensure that the column exists before appending
        #         key = f"ass3_facodec_{i}"
        #         if key in example:
        #             interleaved.append(example[key][j])
        #         else:
        #             # Handle missing columns if necessary
        #             interleaved.append(None)  # or some default value
        # example["ass3_facodec_interleaved"] = interleaved

        return example
    
    # Apply the interleave function to the entire dataset
    interleaved_dataset = dataset.map(interleave)
    return interleaved_dataset


ds_4 = interleave_facodec_columns(ds_1)



def tokenize_and_add_to_dataset(dataset):
    def tokenize_transcript(example):

        if(example["answer"]):
            ass_1_tokenized = tokeniser(example["answer"])['input_ids'] + [end_of_text]
            example["ass_1_tokenized"] = ass_1_tokenized

        else:
            example["ass_1_tokenized"] = None

        # if(example["assistant2"]):
        #     ass_2_tokenized = tokeniser(example["assistant2"])['input_ids'] + [end_of_text]
        #     example["ass_2_tokenized"] = ass_2_tokenized
        # else:
        #     example["ass_2_tokenized"] = None


        # if(example["assistant3"]):
        #     ass_3_tokenized = tokeniser(example["assistant3"])['input_ids'] + [end_of_text]
        #     example["ass_3_tokenized"] = ass_3_tokenized
        # else:
        #     example["ass_3_tokenized"] = None

        if(example["question"]):
            ass_1_tokenized = tokeniser(example["question"])['input_ids'] + [end_of_text]
            example["human_1_tokenized"] = ass_1_tokenized

        else:
            example["human_1_tokenized"] = None

        # if(example["human2"]):
        #     ass_2_tokenized = tokeniser(example["human2"])['input_ids'] + [end_of_text]
        #     example["human_2_tokenized"] = ass_2_tokenized
        # else:
        #     example["human_2_tokenized"] = None


        # if(example["human3"]):
        #     ass_3_tokenized = tokeniser(example["human3"])['input_ids'] + [end_of_text]
        #     example["human_3_tokenized"] = ass_3_tokenized
        # else:
        #     example["human_3_tokenized"] = None
        

        
        return example

    # Apply the tokenization to the dataset
    tokenized_dataset = dataset.map(tokenize_transcript)
    
    return tokenized_dataset

ds_4 = tokenize_and_add_to_dataset(ds_4)

system_message = "You are an AI assistant who will answer the user's questions and follow the user's instructions."


def create_input_ids(example):
    input_ids = [start_of_system] + tokeniser(system_message)["input_ids"] + [end_of_text, end_of_system]

    if example["ass_1_tokenized"]:
        input_ids += [start_of_human] + example["human_1_tokenized"] + [end_of_human] + [start_of_ai] + example["ass_1_tokenized"] + [start_of_speech] + example["ass1_facodec_interleaved"] + [end_of_speech, end_of_ai]
    
    # if example["ass_2_tokenized"]:
    #     input_ids += [start_of_human] + example["human_2_tokenized"] + [end_of_human] + [start_of_ai] + example["ass_2_tokenized"] + [start_of_speech] + example["ass2_facodec_interleaved"] + [end_of_speech, end_of_ai]

    # if example["ass_3_tokenized"]:
    #     input_ids += [start_of_human] + example["human_3_tokenized"] + [end_of_human] + [start_of_ai] + example["ass_3_tokenized"] + [start_of_speech] + example["ass3_facodec_interleaved"] + [end_of_speech, end_of_ai]


    example['input_ids'] = input_ids
    return example


ds_5 = ds_4.map(create_input_ids)

def create_attention_mask(example):
    example['attention_mask'] = [1] * len(example['input_ids'])
    return example

ds_6 = ds_5.map(create_attention_mask)


def preprocess_function(examples):
    examples['labels'] = [
        [(token_id if token_id != pad_token else -100) for token_id in input_ids]
        for input_ids in examples['input_ids']
    ]
    return examples


num_cpus = os.cpu_count()

num_processes = max(1, int(num_cpus * 0.75))

ds_7 = ds_6.map(
    preprocess_function,
    batched=True,
    num_proc=num_processes,
    desc="Preprocessing dataset"
)

#leave only input_ids, attention_mask, and labels

columns_to_keep = ["input_ids", "attention_mask", "labels"]
all_columns = ds_7.column_names
# Identify columns to remove
columns_to_remove = [col for col in all_columns if col not in columns_to_keep]

# Remove unwanted columns
dataset_to_upload = ds_7.remove_columns(columns_to_remove)



dataset_to_upload.push_to_hub(push_name)