import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: str):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    return model, tokenizer


def generate(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
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
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Output:" in response:
        response = response.split("Output:")[-1].strip()
    return response


def chat(model, tokenizer):
    print("\n=== Chat Mode ===")
    print("Type 'quit' to exit\n")

    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() == "quit":
            break

        input_text = input("Input (optional): ").strip()

        response = generate(model, tokenizer, instruction, input_text, top_p=0.9)
        print(f"\nOutput: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned model")
    parser.add_argument("--model_path", type=str, default="./output", help="Model path")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    chat(model, tokenizer)


if __name__ == "__main__":
    main()