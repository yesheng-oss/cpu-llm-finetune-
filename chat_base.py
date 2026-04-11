import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate(model, tokenizer, instruction, input_text="", max_new_tokens=256, temperature=0.7):
    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    else:
        prompt = f"Instruction: {instruction}\nOutput:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Output:" in response:
        response = response.split("Output:")[-1].strip()
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Qwen2-0.5B-Instruct")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    print("\n=== Base Model Chat ===")
    print("Type 'quit' to exit\n")

    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() == "quit":
            break
        input_text = input("Input (optional): ").strip()

        response = generate(model, tokenizer, instruction, input_text)
        print(f"\nOutput: {response}\n")


if __name__ == "__main__":
    main()