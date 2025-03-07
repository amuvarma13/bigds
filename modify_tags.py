from datasets import load_dataset

dsn1 = "amuvarma/luna-48k-5k-snacced"
# dsn2 = "amuvarma/luna-48k-b7CS6GHVkhPt9lmufYchXdy7eLo1-enhanced-clipped-snacced"

ds1 = load_dataset(dsn1, split="train")

ds = ds1

#next make the emotion column all lowercase 
ds = ds.map(lambda x: {"emotion": x["emotion"].lower()})

#next print all unique emotions as a list
unique_emotions = ds.unique("emotion")
print(unique_emotions)
emotion_map = {
    "happy": "happy",
    "normal-longest": "normal",
    "normal-long": "normal",
    "uhsandahs-1": "normal",
    "uhsandahs":"happy",
    "emphasise the word": "normal",
    "sound-ew":"digust", 
    "sound-ew-1":"disgust",
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
    "chuckle": "happy",
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
    "ooh": "surprise",
    "excitement": "excited", 
    "ugh": "disgust",
    "sigh": "normal",
}

#check all emotions are in the map

for emotion in unique_emotions:
    if emotion not in emotion_map:
        print(emotion)

#next replace the emotions with the new values

def map_emotion(row):
    if "emotion" in row:
        if row["emotion"] in emotion_map:
            row["emotion"] = emotion_map[row["emotion"]]
        else:
            row["emotion"] = "normal"
    else:
        row["emotion"] = "normal"
    return row


  

ds = ds.map(map_emotion, num_proc=60, remove_columns=["enhanced_audio"])

ds = ds.shuffle(seed=42).shuffle(seed=42)
print(ds)

unique_emotions = ds.unique("emotion")
print(unique_emotions)


ds.push_to_hub("amuvarma/luna-48k-full-mapped")
