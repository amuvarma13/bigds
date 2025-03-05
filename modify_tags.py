from datasets import load_dataset

dsn1 = "amuvarma/brian-48k-KRVv68cDw2PBeOJypLrzaiI4kol2-enhanced-clipped-snacced"
dsn2 = "amuvarma/luna-48k-b7CS6GHVkhPt9lmufYchXdy7eLo1-enhanced-clipped-snacced"

ds1 = load_dataset(dsn1, split="train")
ds2 = load_dataset(dsn2, split="train")


#first add a column to each dataset to indicate the source brian=>brian and luna=>luna
ds1 = ds1.map(lambda x: {"source": "brian"})
ds2 = ds2.map(lambda x: {"source": "luna"})


print(ds1.column_names)
print(ds2[0]["source"])