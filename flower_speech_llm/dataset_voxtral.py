"""Voxtral dataset and collator for Flower FL.

Builds multimodal prompt + transcription labels using VoxtralProcessor,
following the same pattern as FedEloquence-dev's VoxtralCanonicalCollator.

Supports two data formats:
  1. CSV files (same as the WavLM pipeline) — audio_path + transcript columns
  2. HuggingFace cached datasets — {audio, text, language} schema
"""

import os
import logging
from typing import Dict, List

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Map language names → ISO 639-1 codes for VoxtralProcessor prompts
_LANG_TO_ISO = {
    "english": "en", "french": "fr", "german": "de", "italian": "it",
    "spanish": "es", "portuguese": "pt", "dutch": "nl", "polish": "pl",
    "romanian": "ro", "hungarian": "hu", "czech": "cs", "greek": "el",
    "swedish": "sv", "danish": "da", "finnish": "fi", "norwegian": "no",
    "catalan": "ca", "croatian": "hr", "slovenian": "sl", "slovak": "sk",
    "arabic": "ar", "turkish": "tr", "russian": "ru", "japanese": "ja",
    "korean": "ko", "chinese": "zh", "hindi": "hi", "indonesian": "id",
}


def _to_iso(lang: str) -> str:
    if not lang:
        return "en"
    if len(lang) <= 3:
        return lang
    return _LANG_TO_ISO.get(lang.lower(), lang.lower()[:2])


# ---------------------------------------------------------------------------
# CSV-based dataset (compatible with existing MLS FL partitions)
# ---------------------------------------------------------------------------

class VoxtralCSVDataset(Dataset):
    """Reads a CSV with columns: audio_path, transcript [, language].

    Returns raw dicts for VoxtralCollator to process.
    """

    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row["audio_path"])
        # Resample to 16kHz if needed
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        return {
            "audio": {"array": waveform.squeeze(0).numpy(), "sampling_rate": 16000},
            "text": str(row.get("transcript", row.get("text", ""))),
            "language": str(row.get("language", "en")),
        }


# ---------------------------------------------------------------------------
# Collator — uses VoxtralProcessor to build multimodal inputs
# ---------------------------------------------------------------------------

class VoxtralCollator:
    """Builds multimodal prompt + transcription labels for Voxtral.

    Each sample must contain:
      - audio: dict with "array" (numpy) key
      - text: ground-truth transcription string
      - language: ISO code or language name (optional, defaults to "en")
    """

    def __init__(self, processor, model_id: str, language: str = "en"):
        self.processor = processor
        self.model_id = model_id
        self.language = language

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        audios = [f["audio"]["array"] for f in features]
        per_sample_langs = [_to_iso(f.get("language", "") or self.language)
                            for f in features]

        # Uniform or mixed language argument
        if all(l == per_sample_langs[0] for l in per_sample_langs):
            lang_arg = per_sample_langs[0]
        else:
            lang_arg = per_sample_langs

        # Build multimodal prompt using VoxtralProcessor
        prompt = self.processor.apply_transcription_request(
            language=lang_arg,
            model_id=self.model_id,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )

        # Separate passthrough audio keys from text keys
        passthrough = {
            k: v for k, v in prompt.items()
            if k not in ("input_ids", "attention_mask")
        }

        tok = self.processor.tokenizer
        prompt_ids = prompt["input_ids"]
        prompt_attn = prompt["attention_mask"]

        # Tokenize ground-truth text
        text_tok = tok(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors=None,
        )

        input_ids_list, attention_mask_list, labels_list = [], [], []
        for i, t_ids in enumerate(text_tok["input_ids"]):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()

            ids = p_ids + t_ids + [tok.eos_token_id]
            attn = p_att + [1] * (len(t_ids) + 1)
            lab = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]

            input_ids_list.append(ids)
            attention_mask_list.append(attn)
            labels_list.append(lab)

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids_list)

        def _pad(seq, fill):
            return seq + [fill] * (max_len - len(seq))

        batch = {
            "input_ids": torch.tensor([_pad(x, pad_id) for x in input_ids_list], dtype=torch.long),
            "attention_mask": torch.tensor([_pad(x, 0) for x in attention_mask_list], dtype=torch.long),
            "labels": torch.tensor([_pad(x, -100) for x in labels_list], dtype=torch.long),
        }

        # Passthrough keys carry encoded audio features from the processor
        for k, v in passthrough.items():
            batch[k] = v

        return batch


# ---------------------------------------------------------------------------
# DataLoader builder — mirrors build_dataloaders_from_csvs in dataset.py
# ---------------------------------------------------------------------------

def build_voxtral_dataloaders(csv_dir, processor, model_id, batch_size=2,
                              num_workers=0, shuffle=True, language="en"):
    """Build one DataLoader per CSV file in csv_dir for Voxtral training.

    Args:
        csv_dir: Directory containing CSV files (client_0.csv, client_1.csv, ...).
        processor: VoxtralProcessor instance.
        model_id: Model ID string for apply_transcription_request.
        batch_size: Batch size per DataLoader.
        num_workers: DataLoader workers.
        shuffle: Whether to shuffle.
        language: Default language for prompts.

    Returns:
        List of DataLoader objects.
    """
    collator = VoxtralCollator(processor, model_id, language)

    csv_files = sorted(
        [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]
    )
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {csv_dir}")

    dataloaders = []
    for csv_path in csv_files:
        dataset = VoxtralCSVDataset(csv_path)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=num_workers,
        )
        dataloaders.append(loader)

    logger.info("Created %d Voxtral DataLoaders from '%s'", len(dataloaders), csv_dir)
    return dataloaders
