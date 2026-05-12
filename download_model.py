import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "./models/Qwen2-0.5B-Instruct"

def download_model():
    print(f"Downloading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float32
    )
    
    print(f"Saving to: {OUTPUT_DIR}")
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    download_model()