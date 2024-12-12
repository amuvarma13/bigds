from datasets import load_dataset
dsn = "gpt-omni/VoiceAssistant-400K"
ds = load_dataset(dsn, split='train')
ds = ds.shuffle(seed=42)

ds_0 = ds.select(range(10000))
ds_1 = ds.select(range(10000, 310000))
ds_2 = ds.select(range(310000, 320000))
ds_3 = ds.select(range(320000, 330000))


ds_0.push_to_hub("amuvarma/va-0-10k-snac")
ds_1.push_to_hub("amuvarma/va-10k-310k-snac")
ds_2.push_to_hub("amuvarma/va-310k-320k-snac")
ds_3.push_to_hub("amuvarma/va-320k-330k-snac")



