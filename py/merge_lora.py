import os, shutil
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

BASE = os.environ.get("BASE_MODEL", "t5-small")
LORA = os.environ.get("LORA_DIR", "./t5_creo_lora")
OUT  = os.environ.get("MERGED_DIR", "./t5_creo_merged")

base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, LORA)
model = model.merge_and_unload()  # merges LoRA into base weights

tok = AutoTokenizer.from_pretrained(LORA)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("Merged model saved to:", OUT)
