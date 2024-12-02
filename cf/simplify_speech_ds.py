from datasets import load_dataset
ds = load_dataset("eliasfiz/audio_2m_combined")
new_dataset = ds['train'].select_columns(['transcript', 'facodec_1'])

print(new_dataset) 

first_train = new_dataset.select(range(0, 1000000))
dev = new_dataset.select(range(1000000, 1004000))

dev.push_to_hub("amuvarma/contentonly-raw-dev-4k")
first_train.push_to_hub("amuvarma/contentonly-raw-train-1m")