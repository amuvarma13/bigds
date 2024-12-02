from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import random


dsn = "amuvarma/contentonly-proc-train-1m-1dups"
push_name = "amuvarma/contentonly-proc-train-200k-1dups-concat"

ds = load_dataset(dsn, split='train')

tkn = "meta-llama/Llama-3.2-3B-Instruct"
tokeniser = AutoTokenizer.from_pretrained(tkn)

def read_instructions(filename):
    instructions = []
    with open(filename, 'r') as file:
        instructions = file.read().splitlines()
    return instructions

system_instructs = read_instructions('speech_instructs.txt')

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

start_of_system = tokeniser_length + 8
end_of_system = tokeniser_length + 9

audio_tokens_start = tokeniser_length + 10

def concatenate_rows(dataset):
    # Get all input_ids
    all_input_ids = dataset['input_ids']
    num_complete_groups = len(all_input_ids) // 8
    concatenated_input_ids = []
    
    for i in range(num_complete_groups):
        if(i % 1000 == 0):
            print(f"Processing group {i}")
        # Get 5 rows and concatenate them
        start_idx = i * 8
        group = all_input_ids[start_idx:start_idx + 8]
        concatenated = []
        
        # Add the special tokens at the start
        concatenated.append(start_of_system)
        
        # Tokenize and add random system instruction
        random_instruction = random.choice(system_instructs)
        instruction_tokens = tokeniser(random_instruction)["input_ids"]
        # Remove any special tokens that the tokenizer might have added
        concatenated.extend(instruction_tokens)
        
        concatenated.extend([end_of_text, end_of_system])
        
        # Now add the actual sequences
        for row in group:
            concatenated.extend(row)
            
        concatenated_input_ids.append(concatenated)
    
    # Create new dataset with concatenated rows
    new_dataset = Dataset.from_dict({
        'input_ids': concatenated_input_ids
    })
    
    return new_dataset
# Apply the function to your dataset
new_dataset = concatenate_rows(ds)

print(new_dataset)

new_dataset.push_to_hub(push_name)
