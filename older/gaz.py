from datasets import load_dataset
from eff_pp import preprocess_dataset

dsn = "amuvarma/mls-eng-10k-200k"

ds = load_dataset(dsn)

dsp = preprocess_dataset(ds["train"])

# dspt = dsp.select(range(0,1000))

dsp.push_to_hub("amuvarma/mls-train-200k-1-nopad-pinput-effpreproc")

# dsn1 = "amuvarma/mls-eng-10k-500k"

# ds1 = load_dataset(dsn1)

# dsp1 = preprocess_dataset(ds1["train"])

# dsp1.push_to_hub("amuvarma/mls-train-500")

