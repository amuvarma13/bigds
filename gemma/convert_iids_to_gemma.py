from datasets import load_dataset

dataset = load_dataset("amuvarma/snac-raw-10m", split='train')

print(dataset)