"""LLM wrapper with optional LoRA fine-tuning."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def get_llm(
    model_name: str,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
):
    """Load a causal LLM with tokenizer and optional LoRA adapters.

    Args:
        model_name: HuggingFace model ID (e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0").
        use_lora: Whether to apply LoRA adapters.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.

    Returns:
        Tuple of (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)

    return tokenizer, model
