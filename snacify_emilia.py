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
import requests

sr__ = 24000

def make_request():
    try:
        response = requests.get("http://34.27.188.237:8080/next")
        if response.status_code == 200:
            print(f"Got number: {response.json()['number']}")
        else:
            print(f"Request failed with status {response.status_code}")
    except Exception as e:
        print(f"Error: {str(e)}")



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


def process_tar_file(tar_index):
    path = f"Emilia/EN/EN-B000000.tar"

    if tar_index>1000:
        path = f"Emilia/EN/EN-B00{tar_index}.tar"
    elif tar_index>100:
        path = f"Emilia/EN/EN-B000{tar_index}.tar"
    elif tar_index>10:
        path = f"Emilia/EN/EN-B0000{tar_index}.tar"
    elif tar_index>1:
        path = f"Emilia/EN/EN-B00000{tar_index}.tar"

    
    ds = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")

    ds = ds.map(add_codes, remove_columns=ds.column_names)

    ds.push_to_hub("amuvarma/emilia-snac-1k")




def process_recursively():
    current_index = make_request()
    if current_index > 1139:
        return "ALL PROCESSES DONE"
    
    process_tar_file(current_index)
    process_recursively()
    

process_recursively()

   #step 1 get the index using atomic counter



