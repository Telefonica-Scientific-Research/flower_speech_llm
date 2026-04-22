"""Audio encoder wrappers for SpeechLLM."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor


class TransformerAudioEncoder(nn.Module):
    """Wrapper around a HuggingFace audio encoder (WavLM, HuBERT, Wav2Vec2, etc.)."""

    def __init__(self, model_name: str, finetune: bool = False):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        if not finetune:
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


def get_audio_encoder(encoder_name: str, finetune: bool = False) -> nn.Module:
    """Factory for audio encoders.

    Args:
        encoder_name: HuggingFace model ID or whisper model name.
        finetune: Whether to allow gradients through the encoder.

    Returns:
        An nn.Module audio encoder.
    """
    if "openai/whisper" in encoder_name:
        return WhisperEncoder(encoder_name, finetune)
    else:
        # WavLM, HuBERT, Wav2Vec2, etc.
        return TransformerAudioEncoder(encoder_name, finetune)
