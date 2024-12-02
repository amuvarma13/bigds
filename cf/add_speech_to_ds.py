from datasets import load_dataset, Dataset, Audio
from convert_to_speech import convert_to_speech
import random

dataset_index = 9
dsn = f"amuvarma/smol-talk-everyday-flat"

ds = load_dataset(dsn, split=f"train")
push_name = f"amuvarma/smoltalk-audio-speech_{dataset_index}"


import random

def add_speech_columns(ds):
    def process_row(row):
        try:
            # Create a new dict to avoid modifying the original
            new_row = dict(row)
            speaker_index = random.randint(0, 5)
            
            # Process each assistant response
            for i in range(1, 4):
                assist_key = f'assistant{i}'
                speech_key = f'assistant_speech{i}'
                
                if assist_key in row and row[assist_key] is not None:
                    speech_array = convert_to_speech(row[assist_key], speaker_index)
                    new_row[speech_key] = {
                        'array': speech_array,
                        'sampling_rate': 16000
                    }
                else:
                    new_row[speech_key] = None
                    
            return new_row
        except Exception as e:
            # Optionally, log the error for debugging
            print(f"Skipping row due to error: {e}")
            return None  # Indicate that this row should be skipped

    # Apply the processing function
    processed_ds = ds.map(process_row)

    # Filter out any rows where process_row returned None
    filtered_ds = processed_ds.filter(lambda row: row is not None)

    return filtered_ds


speech_enriched_dataset = add_speech_columns(ds)

for i in range(1, 4):
    speech_key = f'assistant_speech{i}'
    speech_enriched_dataset = speech_enriched_dataset.cast_column(speech_key, Audio())


print(speech_enriched_dataset)

speech_enriched_dataset.push_to_hub(push_name)