import os, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MERGED_DIR = os.environ.get("HF_DIR", "./t5_creo_merged")
OUT_DIR    = os.environ.get("ONNX_DIR", "./onnx_model")
os.makedirs(OUT_DIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MERGED_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MERGED_DIR)
model.eval()

# Dummy example
enc = tok("Create a 50 mm cube using rectangle sketch and extrude.", return_tensors="pt")
dec = tok([""], return_tensors="pt")  # decoder starts with <pad> token (T5 uses pad as start)

# Forward signature: (input_ids, attention_mask, decoder_input_ids) -> logits
class T5Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        return out.logits

wrapper = T5Wrapper(model)

torch.onnx.export(
    wrapper,
    (enc["input_ids"], enc["attention_mask"], dec["input_ids"]),
    f"{OUT_DIR}/t5_creo.onnx",
    input_names=["input_ids", "attention_mask", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0:"batch", 1:"src_seq"},
        "attention_mask": {0:"batch", 1:"src_seq"},
        "decoder_input_ids": {0:"batch", 1:"tgt_seq"},
        "logits": {0:"batch", 1:"tgt_seq", 2:"vocab"}
    },
    opset_version=17
)

# Save tokenizer artifacts for C++
tok.save_pretrained(OUT_DIR)  # includes spiece.model for T5
print("Exported ONNX model + tokenizer to:", OUT_DIR)

