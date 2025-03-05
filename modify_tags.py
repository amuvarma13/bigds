from datasets import load_dataset

dsn1 = "amuvarma/brian-48k-KRVv68cDw2PBeOJypLrzaiI4kol2-enhanced-clipped-snacced"
dsn2 = "amuvarma/luna-48k-b7CS6GHVkhPt9lmufYchXdy7eLo1-enhanced-clipped-snacced"

ds1 = load_dataset(dsn1, split="train")
ds2 = load_dataset(dsn2, split="train")


#first add a column to each dataset to indicate the source brian=>brian and luna=>luna
ds1 = ds1.map(lambda x: {"source": "brian"})
ds2 = ds2.map(lambda x: {"source": "luna"})

#next merge the datasets
from datasets import concatenate_datasets

ds = concatenate_datasets([ds1, ds2])

#next make the emotion column all lowercase 
ds = ds.map(lambda x: {"emotion": x["emotion"].lower()})

#next print all unique emotions as a list
unique_emotions = ds.unique("emotion")
print(unique_emotions)

emotion_map = {
    "happy": "happy",
    "normally": "normal",
    "longer": "longer",
    "sad": "sad",
    "frustrated": "frustrated",
    "pausing": "normal",
    "read slowly": "slow",
    "disgust": "disgust",
    "excited": "excited",
    "whisper": "whisper",
    "panicky": "panicky",
    "curious": "curious",
    "surprise": "surprise",
    "chuckle": "chuckle",
    "slow": "slow",
    "fast": "fast",
    "crying": "crying",
    "with a deep voice": "deep",
    "sleepy": "sleepy",
    "angry": "angry",
    "normal": "normal",
    "hmm": "curious",
    "with a high pitched voice": "high",
    "shout": "shout",
    "emphasise the word": "normal",
    "ooh": "ooh",
    "excitement": "excited"
}

#next replace the emotions with the new values

def map_emotion(row):
    print(row["emotion"])
    if "emotion" in row:
        if row["emotion"] in emotion_map:
            row["emotion"] = emotion_map[row["emotion"]]
 
        else:
            row["emotion"] = "normal"
        return row
    else:
        row["emotion"] = "normal"
        return row

ds = ds.map(map_emotion, num_proc=60)

print(ds)