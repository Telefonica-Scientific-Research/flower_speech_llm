import torch
from transformers import AutoProcessor, AutoFeatureExtractor
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import random
import numpy as np


class MyCollator:
    def __init__(self, audio_encoder_name, tokenizer):
        self.audio_encoder_name = audio_encoder_name
        self.tokenizer = tokenizer
        self.hubert_processor = AutoFeatureExtractor.from_pretrained(
            "microsoft/wavlm-base")  # change according to the encoder

    def _extract_features(self, waveform):
        """Extract audio features from a raw waveform tensor."""
        if waveform is None:
            return None
        if "openai/whisper" in self.audio_encoder_name:
            return self.wav_2_mel(waveform).unsqueeze(0)
        return self.hubert_processor(
            waveform.squeeze(), return_tensors="pt", sampling_rate=16000
        ).input_values

    def __call__(self, batch):
        """Collate a batch of samples with proper padding for batch_size >= 1.

        Returns a tuple of 8 tensors:
            mel:      (B, max_audio_len) — padded audio features
            mel_mask: (B, max_audio_len) — 1 for real frames, 0 for padding
            pre_ids:  (B, max_pre_len)   — padded pre-prompt token ids
            pre_mask: (B, max_pre_len)   — attention mask
            post_ids: (B, max_post_len)  — padded post-prompt token ids
            post_mask:(B, max_post_len)  — attention mask
            out_ids:  (B, max_out_len)   — padded output token ids
            out_mask: (B, max_out_len)   — attention mask
        """
        waveforms, pre_prompts, post_prompts, output_prompts = [], [], [], []
        for sample in batch:
            waveform, pre, post, output, _complete = sample
            waveforms.append(waveform.squeeze().numpy() if waveform is not None else None)
            pre_prompts.append(pre)
            post_prompts.append(post)
            output_prompts.append(output)

        # ---- Audio features (padded) ----
        if waveforms[0] is not None:
            if "openai/whisper" in self.audio_encoder_name:
                # Whisper: mel spectrogram (fixed length per chunk)
                import whisper
                mels = [self.wav_2_mel(torch.from_numpy(w).unsqueeze(0)) for w in waveforms]
                mel = torch.stack(mels, dim=0)
                mel_mask = torch.ones(mel.shape[:2], dtype=torch.long)
            else:
                # WavLM / HuBERT: AutoFeatureExtractor handles padding
                features = self.hubert_processor(
                    waveforms,
                    return_tensors="pt",
                    sampling_rate=16000,
                    padding=True,
                    return_attention_mask=True,
                )
                mel = features.input_values        # (B, max_audio_len)
                mel_mask = features.attention_mask  # (B, max_audio_len)
        else:
            mel, mel_mask = None, None

        # ---- Text tokens (padded) ----
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        pre_enc = self.tokenizer(
            pre_prompts, padding=True, return_tensors='pt',
            truncation=False, add_special_tokens=False,
        )
        post_enc = self.tokenizer(
            post_prompts, padding=True, return_tensors='pt',
            truncation=False, add_special_tokens=False,
        )
        output_texts = [
            self.tokenizer.bos_token + o + self.tokenizer.eos_token
            for o in output_prompts
        ]
        out_enc = self.tokenizer(
            output_texts, padding=True, return_tensors='pt',
            truncation=False, add_special_tokens=False,
        )

        return (mel, mel_mask,
                pre_enc.input_ids, pre_enc.attention_mask,
                post_enc.input_ids, post_enc.attention_mask,
                out_enc.input_ids, out_enc.attention_mask)

    def wav_2_mel(self, wav_tensor):
        mel = whisper.log_mel_spectrogram(wav_tensor[0])
        return mel


class AudioDataset(Dataset):
    def __init__(self, csv_file, mode='train', random_keys_prob=0.001):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
        self.mode = mode
        self.random_keys_prob = random_keys_prob
        self.labels = ['isspeech', 'transcript', 'gender', 'emotion', 'age', 'accent']

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Load audio
        audio_row = self.data_frame.iloc[idx]
        audio_path = audio_row['audio_path']
        if pd.isna(audio_path):
            waveform = None
        else:
            waveform, sample_rate = torchaudio.load(audio_path)

        # Prepare labels dictionary based on mode and probability
        labels_str = {}
        if self.mode == 'train' and random.random() < self.random_keys_prob:
            random_labels = random.sample(self.labels, k=random.randint(1, len(self.labels)))
            for label in random_labels:
                if label in audio_row and pd.notnull(audio_row[label]):
                    formatted_label = label.capitalize()
                    if audio_row[label] == True or audio_row[label] == False:
                        labels_str[formatted_label] = audio_row[label]
                    else:
                        labels_str[formatted_label] = str(audio_row[label]).lower()
        else:
            # Most of the time, include all available labels
            for label in self.labels:
                if label in audio_row and pd.notnull(audio_row[label]):
                    formatted_label = label.capitalize()
                    if audio_row[label] == True or audio_row[label] == False:
                        labels_str[formatted_label] = audio_row[label]
                    else:
                        labels_str[formatted_label] = str(audio_row[label]).lower()

        if 'context' in audio_row.index:
            conv_history = audio_row['context']
        else:
            conv_history = ""

        return waveform, labels_str, conv_history


class InstructionalAudioDataset(AudioDataset):
    def __init__(self, csv_file, mode='train', random_keys_prob=0.1):
        """
        Initialize the class with the specified CSV file, mode, and random keys probability.

        Args:
            csv_file (str): The path to the CSV file.
            mode (str, optional): The mode of the operation, defaults to 'train'.
            random_keys_prob (float, optional): The probability of using random keys, defaults to 0.1.

        Returns:
            None
        """
        super().__init__(csv_file, mode, random_keys_prob)
        self.instruction_phrases = [
            "Provide the details about the audio",
            "I need the following information from the audio",
            "Tell me about the audio regarding",
            "Extract the following details from the audio",
            "Give me the following information about the audio",
            "Provide details from the audio file",
            "I need information extracted from this speech",
            "Detail the contents of the following audio",
            "Share insights about this speech recording",
            "Describe the specifics captured in this audio file",
            "Summarize the audio's key information",
            "Convey the details embedded in this speech",
            "Outline the main points from this audio file",
            "Unpack the content of the following speech",
            "Present the facts from this audio recording",
            "Elucidate the elements within this speech",
            "Decipher the audio file's information",
            "Break down the details in this speech",
            "Analyze the following audio for details",
            "Report on the specifics of this speech file",
            "Transcribe the key points from this audio",
            "Explain the content of the speech recording",
            "Interpret the information within this audio file",
            "Catalog the details from this speech",
            "Narrate the findings in the audio",
            "Recount the specifics of this speech file",
            "Review the contents of the audio",
            "Assess the information provided by this speech",
            "Evaluate the details in the audio file",
            "Investigate the speech for key information",
            "Scrutinize the audio and provide insights",
            "Inspect the details within this speech",
            "Examine the audio file for specific information",
            "Survey the speech and detail your findings",
            "Study the audio and summarize the content",
            "Audit the speech for important details",
            "Appraise the audio file's key points",
            "Annotate the specifics found in the speech",
            "Dissect the audio to find important information",
            "Extract insights from the speech file",
            "Unveil the details in the audio recording",
            "Shed light on the speech's content",
            "Clarify the specifics within the audio file",
            "Illuminate the information in the speech",
            "Highlight the key points of the audio",
            "Reveal the contents captured in the speech file",
            "Uncover the details within the audio",
            "Delve into the speech for essential information",
            "Probe the audio file for details",
            "Explore the speech recording's specifics",
            "Research the contents of the audio",
            "Inquire into the details of the speech",
            "Sift through the audio for key information",
            "Dive into the speech to extract details",
            "Investigate the nuances of the audio file",
            "Give me the following information about the audio",
            "Fetch information",
            "Give me details about the audio",
            "what does this audio say",
            'what is in the file',
            'give me these details',
        ]

    def __getitem__(self, idx):
        waveform, labels_str, conv_history = super().__getitem__(idx)
        instruction_phrase = random.choice(self.instruction_phrases)

        pre_speech_prompt = f"Instruction:\n{instruction_phrase} - ["
        pre_speech_prompt += ', '.join(
            ['IsSpeech' if k == 'isSpeech' else k for k in labels_str.keys()]) + "]\n\nInput:\n<speech>"
        pre_speech_prompt = pre_speech_prompt.replace("Isspeech", "SpeechActivity")
        post_speech_prompt = f"</speech>\n\n" + \
                             "Output:\n"
        output_prompt = "{"
        for key, value in labels_str.items():
            if key == "Isspeech": key = 'SpeechActivity'
            output_prompt += f'  "{key}": "{value}", '
        output_prompt = output_prompt.rstrip(',\n') + "}"

        complete_prompt = pre_speech_prompt + post_speech_prompt + output_prompt
        return waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt


def build_dataloaders_from_csvs(csv_dir, my_collator, batch_size=1, num_workers=3, shuffle=True):
    """
    Load all CSV files in a directory, create a DataLoader for each,
    and return them in a list.

    Args:
        csv_dir (str): Directory containing CSV files (e.g., client_1.csv, client_2.csv, ...).
        my_collator: Your custom collator object.
        batch_size (int): Batch size for each DataLoader.
        num_workers (int): Number of worker processes for loading data.
        shuffle (bool): Whether to shuffle each dataset.

    Returns:
        list: List of DataLoader objects.
    """
    dataloaders = []

    # Get all CSV files (sorted for consistent order)
    csv_files = sorted(
        [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]
    )

    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {csv_dir}")

    # Build one dataset + dataloader per CSV file
    for csv_path in csv_files:
        dataset = InstructionalAudioDataset(csv_file=csv_path, mode='train', random_keys_prob=0.1)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=my_collator,
            num_workers=num_workers,
        )

        dataloaders.append(loader)

    print(f"✅ Created {len(dataloaders)} DataLoaders from {len(csv_files)} CSV files in '{csv_dir}'")
    return dataloaders


# Example usage
if __name__ == "__main__":
    dataset = InstructionalAudioDataset(csv_file='dev.csv', mode='test', random_keys_prob=0.0001)
    waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt = dataset[121]

    print(complete_prompt)
    print(waveform)

    my_collator = MyCollator("microsoft/wavlm-large", tokenizer=None)  # example placeholder

    dataloaders = build_dataloaders_from_csvs(
        csv_dir="./fl_partitions_speaker",
        my_collator=my_collator,
        batch_size=1,
        num_workers=3,
        shuffle=True
    )

    print(f"Total DataLoaders: {len(dataloaders)}")
