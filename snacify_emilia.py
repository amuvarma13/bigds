from datasets import load_dataset

from huggingface_hub import snapshot_download
from datasets import load_dataset
import torch
import random

repo_id = "amphion/Emilia-Dataset"

from snac import SNAC

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
model = model.to("cuda")
import torchaudio.transforms as T






def tokenise_audio(waveform):
  waveform = torch.from_numpy(waveform).unsqueeze(0)
  waveform = waveform.to(dtype=torch.float32)


  waveform = waveform.unsqueeze(0).to("cuda")
  #generate the codes from snac
  with torch.inference_mode():
    codes = model.encode(waveform)

  all_codes = []
  for i in range(codes[0].shape[1]):
    all_codes.append(codes[0][0][i].item()+128266)
    all_codes.append(codes[1][0][2*i].item()+128266+4096)
    all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
    all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
    all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
    all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
    all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))


  return all_codes


def add_codes(example):
    # Always initialize codes_list to None
    codes_list = None

    try:
        answer_audio = example.get("mp3")
        # If there's a valid audio array, tokenise it
        if answer_audio and "array" in answer_audio:
            audio_array = answer_audio["array"]
            codes_list = tokenise_audio(audio_array)
    except Exception as e:
        print(f"Skipping row due to error: {e}")
        # Keep codes_list as None if we fail



    example["codes_list"] = codes_list
    example["text"] = example["json"]["text"]

    return example

path = "Emilia/EN/EN-B000000.tar"
ds = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")
sr__ = 24000
ds = ds.map(add_codes, remove_columns=ds.column_names)


ds.push_to_hub("amuvarma/emilia-snac-1k")