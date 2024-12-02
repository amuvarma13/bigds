import random
import multiprocessing
from datasets import load_dataset

file_path = 'transcribe_exps.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        expressions = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    raise FileNotFoundError(f"The file {file_path} does not exist.")
except IOError:
    raise IOError(f"An error occurred while reading the file {file_path}.")

dsn = "amuvarma/mls-eng-10k-500k"
push_name = "amuvarma/mls-eng-10k-500k-projection_prep"

ds = load_dataset(dsn, split="train")

num_proc = min(multiprocessing.cpu_count(), 16)


def transform_batch(batch):
    user_messages = [random.choice(expressions) for _ in batch['transcript']]
    assistant_messages = [
        (t.strip().capitalize() + '.') if t.strip() else '.' for t in batch['transcript']
    ]
    return {
        'audio': batch['audio'],
        'user': user_messages,
        'assistant': assistant_messages
    }

transformed_ds = ds.map(
    transform_batch,
    batched=True,
    batch_size=1000,
    num_proc=num_proc,
    remove_columns=[col for col in ds.column_names if col not in ['audio', 'transcript']]
)

transformed_ds = transformed_ds.shuffle(seed=42)

transformed_ds.push_to_hub(push_name)