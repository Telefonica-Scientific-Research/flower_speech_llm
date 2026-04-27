"""Audio encoder wrappers for SpeechLLM."""

import torch
import torch.nn as nn
import warnings
from transformers import AutoModel, AutoFeatureExtractor
from peft import LoraConfig, get_peft_model


class TransformerAudioEncoder(nn.Module):
    """Wrapper around a HuggingFace audio encoder (WavLM, HuBERT, Wav2Vec2, etc.)."""

    def __init__(self, model_name: str, finetune: bool = False,
                 use_lora: bool = False, lora_r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        if finetune and use_lora:
            # Freeze all, then apply LoRA on attention projections
            for param in self.model.parameters():
                param.requires_grad = False
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj"],
                bias="none",
                inference_mode=False,
            )
            self.model = get_peft_model(self.model, lora_config)
        elif finetune and not use_lora:
            warnings.warn(
                f"Full fine-tuning of encoder '{model_name}' — all weights are "
                "trainable. This is memory-intensive and increases FL communication "
                "cost. Consider use_lora=True for parameter-efficient training.",
                stacklevel=2,
            )
        elif not finetune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input features from the feature extractor, shape (B, seq_len) or (B, 1, seq_len).
            attention_mask: Optional (B, seq_len), 1 for real frames, 0 for padding.
        Returns:
            Hidden states from the last layer, shape (B, T, D).
        """
        outputs = self.model(x, attention_mask=attention_mask)
        # Use last_hidden_state if available, otherwise hidden_states
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[0]


class WhisperEncoder(nn.Module):
    """Wrapper around OpenAI Whisper encoder."""

    def __init__(self, model_name: str, finetune: bool = False):
        super().__init__()
        import whisper
        size = model_name.split("-")[-1] if "-" in model_name else "large-v3-turbo"
        self.model = whisper.load_model(size).encoder
        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)


def get_audio_encoder(encoder_name: str, finetune: bool = False,
                      use_lora: bool = False, lora_r: int = 8,
                      lora_alpha: int = 16) -> nn.Module:
    """Factory for audio encoders.

    Args:
        encoder_name: HuggingFace model ID or whisper model name.
        finetune: Whether to allow gradients through the encoder.
        use_lora: If True and finetune is True, apply LoRA instead of full unfreeze.
        lora_r: LoRA rank (only used when use_lora=True).
        lora_alpha: LoRA alpha scaling (only used when use_lora=True).

    Returns:
        An nn.Module audio encoder.
    """
    if "openai/whisper" in encoder_name:
        return WhisperEncoder(encoder_name, finetune)
    else:
        # WavLM, HuBERT, Wav2Vec2, etc.
        return TransformerAudioEncoder(encoder_name, finetune, use_lora, lora_r, lora_alpha)
