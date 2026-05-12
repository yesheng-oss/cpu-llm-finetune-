import os
import argparse
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    max_seq_length: int = 512


@dataclass
class TrainConfig:
    output_dir: str = "./output"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2


def setup_model(model_config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.config.use_cache = False

    return model, tokenizer


def setup_lora():
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    return lora_config


def preprocess_function(examples, tokenizer, max_length):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        if input_text:
            text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
        else:
            text = f"Instruction: {instruction}\nOutput: {output}"
        texts.append(text)

    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_train_data(data_path: str, tokenizer, max_length: int):
    dataset = load_dataset("json", data_files=data_path, split="train")

    def preprocess(examples):
        return preprocess_function(examples, tokenizer, max_length)

    dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing dataset",
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on CPU")
    parser.add_argument("--model_name", type=str, default=ModelConfig.model_name)
    parser.add_argument("--data_path", type=str, default="./data/train.jsonl", help="Training data path")
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--num_epochs", type=int, default=TrainConfig.num_train_epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.per_device_train_batch_size)
    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--max_length", type=int, default=ModelConfig.max_seq_length)
    args = parser.parse_args()

    model_config = ModelConfig(model_name=args.model_name, max_seq_length=args.max_length)
    train_config = TrainConfig(output_dir=args.output_dir)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = setup_model(model_config)

    print("Setting up LoRA...")
    lora_config = setup_lora()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading dataset from: {args.data_path}")
    train_dataset = load_train_data(args.data_path, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=TrainConfig.warmup_steps,
        logging_steps=TrainConfig.logging_steps,
        save_steps=TrainConfig.save_steps,
        save_total_limit=TrainConfig.save_total_limit,
        fp16=False,
        report_to=["none"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()