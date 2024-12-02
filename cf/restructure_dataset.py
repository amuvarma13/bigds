dsn = "HuggingFaceTB/smoltalk"

from datasets import load_dataset, Dataset

ds = load_dataset(dsn, split="train")

def flatten_conversations(ds):

    flattened_data = {
        'id': [],
        'human1': [], 'assistant1': [],
        'human2': [], 'assistant2': [],
        'human3': [], 'assistant3': []
    }
    
    for item in ds:
        conversation = item['data']
        conv_id = item['id']
        
        # Initialize all fields as None
        current_item = {
            'id': conv_id,
            'human1': None, 'assistant1': None,
            'human2': None, 'assistant2': None,
            'human3': None, 'assistant3': None
        }
        
        # Populate available turns
        for i, turn in enumerate(conversation):
            if i % 2 == 0:  # Human turns (even indices)
                human_num = (i // 2) + 1
                if human_num <= 3:  # Only process up to human3
                    current_item[f'human{human_num}'] = turn
            else:  # Assistant turns (odd indices)
                assistant_num = (i // 2) + 1
                if assistant_num <= 3:  # Only process up to assistant3
                    current_item[f'assistant{assistant_num}'] = turn
        
        # Add the current item to flattened data
        for key in flattened_data:
            flattened_data[key].append(current_item[key])
    
    # Convert to HuggingFace Dataset
    from datasets import Dataset
    return Dataset.from_dict(flattened_data)

flattened_ds = flatten_conversations(ds)
def filter_long_responses(ds, max_length=700):
    def is_valid_row(row):
        # Check each assistant response
        for i in range(1, 4):
            assist_key = f'assistant{i}'
            if assist_key in row and isinstance(row[assist_key], str):
                if len(row[assist_key]) > max_length:
                    return False
        return True
    
    # Filter the dataset
    filtered_ds = ds.filter(is_valid_row)
    
    return filtered_ds
flattened_ds = filter_long_responses(flattened_ds)
flattened_ds.push_to_hub("amuvarma/ultrachat-25k-audio-flattened")
flattened_ds_dev = flattened_ds.select(range(10))

flattened_ds_dev.push_to_hub("amuvarma/ultrachat-25k-audio-flattened-dev")