"""Voxtral model loader with LoRA fine-tuning for Flower FL.

Supports three training modes controlled by config flags:
  - connector (multi_modal_projector): always fully trainable
  - LLM (language_model): LoRA on Q/K/V/O projections
  - audio encoder (audio_tower): frozen by default, LoRA when finetune_encoder=True
"""

import torch
from huggingface_hub import snapshot_download
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor
from peft import LoraConfig, get_peft_model


def _resolve_local_model_path(model_name: str, cache_dir: str = "") -> str:
    """Resolve the local snapshot path for a HuggingFace model.

    Uses snapshot_download with local_files_only=True to find the cached
    model directory without any network access.  Falls back to the original
    model_name (works when online).
    """
    dl_kwargs = {}
    if cache_dir:
        dl_kwargs["cache_dir"] = cache_dir
    try:
        return snapshot_download(model_name, local_files_only=True, **dl_kwargs)
    except Exception:
        return model_name


def get_voxtral(
    model_name: str = "mistralai/Voxtral-Mini-3B-2507",
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 32,
    finetune_encoder: bool = False,
    cache_dir: str = "",
):
    """Load Voxtral with optional LoRA adapters.

    Training targets:
      1. multi_modal_projector — fully trainable (no LoRA, all weights exchanged in FL)
      2. language_model — LoRA on attention projections
      3. audio_tower — frozen by default; LoRA on attention projections when
         finetune_encoder=True

    Args:
        model_name: HuggingFace model ID.
        use_lora: Whether to apply LoRA adapters on the language model.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        finetune_encoder: If True, also apply LoRA to the audio_tower.
        cache_dir: Optional HF cache directory.

    Returns:
        Tuple of (processor, model) where model has the correct requires_grad settings.
    """
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    # Resolve to local snapshot path so mistral_common's tokenizer loader
    # doesn't try to hit the HF API (fails on offline HPC compute nodes).
    local_path = _resolve_local_model_path(model_name, cache_dir)

    # Load model in bfloat16 for memory efficiency
    model = VoxtralForConditionalGeneration.from_pretrained(
        local_path, torch_dtype=torch.bfloat16, **kwargs
    )
    processor = VoxtralProcessor.from_pretrained(local_path, **kwargs)

    # Step 1: Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: Unfreeze the connector (multi_modal_projector) — full training
    for p in model.multi_modal_projector.parameters():
        p.requires_grad = True

    # Step 3: Apply LoRA
    if use_lora:
        # Build target_modules based on what we want to LoRA-train
        target_modules = ["language_model.*q_proj", "language_model.*k_proj",
                          "language_model.*v_proj", "language_model.*o_proj"]
        if finetune_encoder:
            target_modules += ["audio_tower.*q_proj", "audio_tower.*k_proj",
                               "audio_tower.*v_proj", "audio_tower.*o_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return processor, model
