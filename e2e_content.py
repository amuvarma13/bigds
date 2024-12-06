from datasets import load_dataset
dsn = "eliasfiz/qa_large_0_4_speechqa-both-full-answer-facodec"
ds = load_dataset(dsn, split = "train")
ds1 = ds.select(range(10000))
ds2 = ds.select(range(10000,20000))
ds1.push_to_hub("amuvarma/audio-in-out-10k_part1")
ds2.push_to_hub("amuvarma/audio-in-out-10k_part2")
