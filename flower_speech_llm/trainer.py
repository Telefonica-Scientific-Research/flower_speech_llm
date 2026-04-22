import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import wandb
import pytorch_lightning as pl
import numpy as np
from jiwer import wer
import torchmetrics
import random
import re
import json

from collections import OrderedDict

from .model.encoder import get_audio_encoder, TransformerAudioEncoder
from .model.connector import get_connector, LinearConnector, LinearPoolConnector, CNNConnector
from .model.llm import get_llm


def save_trainable_state_dict(model, path):
    """Save only trainable parameters (LoRA adapters + connector).

    This produces a checkpoint ~100x smaller than the full state_dict
    since frozen encoder and LLM base weights are excluded.
    """
    state_dict = OrderedDict(
        {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    )
    torch.save(state_dict, path)
    return state_dict


def load_trainable_state_dict(model, path):
    """Load a trainable-only checkpoint (LoRA + connector) into a full model.

    Non-trainable (frozen) weights are kept as initialized from pretrained
    HuggingFace models.  Works with both adapter-only and full state_dicts.
    """
    state_dict = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Only warn about truly unexpected keys (not just frozen base weights)
    if unexpected:
        print(f"  ⚠️  {len(unexpected)} unexpected keys in checkpoint (ignored)")
    return model


class SpeechLLMLightning(pl.LightningModule):
    def __init__(self, 
                 audio_enc_dim=512, 
                 llm_dim=2048, 
                 audio_encoder_name="speech-tokenizer",
                 connector_name='linear-pool',
                 llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 finetune_encoder=False,
                 connector_k=5,
                 use_lora=True,
                 lora_r=32,
                 lora_alpha=2,
                 max_lr=3e-4,
                 total_training_step=500000,
                 warmup_steps=1000,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.audio_enc_dim = audio_enc_dim
        self.llm_dim = llm_dim
        self.llm_name = llm_name
        self.finetune_encoder = finetune_encoder
        self.use_lora = use_lora

        self.audio_encoder = get_audio_encoder(audio_encoder_name, finetune_encoder)
        self.connector = get_connector(connector_name, audio_enc_dim, llm_dim, connector_k)
        self.llm_tokenizer, self.llm_model = get_llm(llm_name, use_lora, lora_r, lora_alpha)
        
        self.max_lr = max_lr
        self.total_training_step = total_training_step
        self.warmup_steps = warmup_steps
        self.use_embedding_loss = False
        self.num_validation_samples = 5000

    def configure_optimizers(self):
        opt = [
            {"params": self.audio_encoder.parameters(), "lr": 1e-5},
            {"params": self.connector.parameters(), "lr": self.max_lr},
            {"params": self.llm_model.parameters(), "lr": self.max_lr},
        ]
        optimizer = Adam(opt, lr=self.max_lr)
        return optimizer

    def encode(self, mel, mel_mask, pre_ids, pre_mask, post_ids, post_mask,
               out_ids, out_mask, return_embedding_loss=False):
        """Build combined embeddings, attention mask, and labels for the LLM.

        Supports batch_size >= 1 with proper padding masks.  The combined
        sequence is: [pre_prompt | speech_embeds | post_prompt | output].
        Padding positions in any segment are masked in the attention mask and
        get label_id = -100 so they don't contribute to the loss.
        """
        batch_size = mel.shape[0]

        # ---- Audio encoder (with padding mask) ----
        speech_embeds = self.audio_encoder(mel, attention_mask=mel_mask)
        speech_embeds = self.connector(speech_embeds)
        # Compute valid speech-embed lengths from audio input lengths
        input_lengths = mel_mask.sum(dim=1)  # (B,)
        enc_output_lengths = self.audio_encoder.model._get_feat_extract_output_lengths(input_lengths)
        # Build speech attention mask: 1 for valid encoder frames, 0 for padding
        speech_seq_len = speech_embeds.shape[1]
        speech_mask = (torch.arange(speech_seq_len, device=mel.device)[None, :]
                       < enc_output_lengths[:, None]).long()

        # ---- Text embeddings ----
        embedder = self.llm_model.get_input_embeddings()
        pre_prompt_embeds = embedder(pre_ids)
        post_prompt_embeds = embedder(post_ids)
        output_prompt_embeds = embedder(out_ids)

        # ---- Concatenate along sequence dimension ----
        combined_embeds = torch.cat(
            [pre_prompt_embeds, speech_embeds, post_prompt_embeds, output_prompt_embeds], dim=1)

        # ---- Combined attention mask ----
        combined_atts = torch.cat(
            [pre_mask, speech_mask, post_mask, out_mask], dim=1
        ).to(combined_embeds.device)

        # ---- Labels: -100 for input/speech/post positions AND padding ----
        input_token_length = pre_ids.shape[1] + speech_embeds.shape[1] + post_ids.shape[1]
        input_labels = torch.full(
            [batch_size, input_token_length], -100,
            device=combined_embeds.device, dtype=torch.int64)
        output_labels = out_ids.clone().to(combined_embeds.device).to(torch.int64)
        # Mask out padding positions in output
        output_labels[out_mask == 0] = -100
        label_ids = torch.cat([input_labels, output_labels], dim=1)

        return combined_embeds, combined_atts, label_ids

    def forward(self, embeds, atts, label_ids):
        out = self.llm_model(
            inputs_embeds=embeds,
            attention_mask=atts,
            labels=label_ids,
        )
        return out
    
    def training_step(self, batch, batch_idx):
        mel, mel_mask, pre_ids, pre_mask, post_ids, post_mask, out_ids, out_mask = batch
        embeds, atts, label_ids = self.encode(
            mel, mel_mask, pre_ids, pre_mask, post_ids, post_mask, out_ids, out_mask)
        outputs = self.forward(embeds, atts, label_ids)
        loss =  outputs["loss"]
        self.log("train/loss", loss, on_epoch=False)
        return loss
    
    def _decode_output_only(self, predicted_ids, label_ids):
        """Decode only the output-portion predictions (where labels != -100).

        Teacher-forced argmax over the full sequence produces garbage for
        input / speech-embedding positions.  Decoding only the output
        portion yields a much cleaner string for metric extraction.
        """
        label_mask = (label_ids[0] != -100).cpu()
        output_pred_ids = predicted_ids[0].cpu()[label_mask]
        return self.llm_tokenizer.decode(output_pred_ids, skip_special_tokens=True)

    @staticmethod
    def _extract_field_robust(text, field_name):
        """Extract a field value from (possibly malformed) JSON-like text."""
        pattern = rf'"{field_name}"\s*:\s*"([^"]*)"'
        match = re.search(pattern, text)
        return match.group(1) if match else ""

    def _extract_all_fields_robust(self, text):
        """Extract all known fields via robust regex (no JSON parsing)."""
        fields = {}
        for field in ("Transcript", "Response", "SpeechActivity", "Gender",
                       "Emotion", "Age", "Accent"):
            val = self._extract_field_robust(text, field)
            if val:
                fields[field] = val
        return fields

    def validation_step(self, batch, batch_idx):
            mel, mel_mask, pre_ids, pre_mask, post_ids, post_mask, out_ids, out_mask = batch
            embeds, atts, label_ids = self.encode(
                mel, mel_mask, pre_ids, pre_mask, post_ids, post_mask, out_ids, out_mask)
            outputs = self.forward(embeds, atts, label_ids)
            loss = outputs["loss"]
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode only output-portion predictions to avoid input-position noise
            generated_output_text = self._decode_output_only(predicted_ids, label_ids)
            target_text = self.llm_tokenizer.decode(out_ids[0], skip_special_tokens=True)
            
            # Robust field extraction (tolerates malformed JSON from teacher forcing)
            extracted_pred = self._extract_all_fields_robust(generated_output_text)
            extracted_target = self._extract_all_fields_robust(target_text)

            keys = extracted_target.keys()

            for key in keys:
                if key not in extracted_pred:
                    extracted_pred[key] = "NA"

            if 'Transcript' in keys:
                target_transcript = extracted_target['Transcript']
                predicted_transcript = extracted_pred['Transcript']
                wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
                self.log("val/wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Response' in keys:
                target_transcript = extracted_target['Response']
                predicted_transcript = extracted_pred['Response']
                wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
                self.log("val/response_wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'SpeechActivity' in keys:
                target_isspeech = extracted_target['SpeechActivity']
                predicted_isspeech = extracted_pred['SpeechActivity']
                self.log("val/speech_activity", float(target_isspeech.lower()==predicted_isspeech.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Gender' in keys:
                target_gender = extracted_target['Gender']
                predicted_gender = extracted_pred['Gender']
                self.log("val/gender", float(target_gender.lower()==predicted_gender.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Emotion' in keys:
                target_emotion = extracted_target['Emotion']
                predicted_emotion = extracted_pred['Emotion']
                self.log("val/emotion", float(target_emotion.lower()==predicted_emotion.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Age' in keys:
                target_age = extracted_target['Age']
                predicted_age = extracted_pred['Age']
                self.log("val/age", float(target_age.lower()==predicted_age.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Accent' in keys:
                target_accent = extracted_target['Accent']
                predicted_accent = extracted_pred['Accent']
                self.log("val/accent", float(target_accent.lower()==predicted_accent.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if batch_idx in self.selected_samples_for_logging:
                sample_idx = self.selected_samples_for_logging.index(batch_idx)
                # Use wandb.log to log prediction and truth texts
                wandb.log({
                    f"val_sample_{sample_idx}_pred": wandb.Html(f"<pre>{str(extracted_pred)}</pre>"), 
                    f"val_sample_{sample_idx}_target": wandb.Html(f"<pre>{str(extracted_target)}</pre>"),
                    f"val_sample_{sample_idx}_gen": wandb.Html(f"<pre>{generated_output_text}</pre>"),
                }, commit=False)

            return {"val_loss": loss}
    
    def test_step(self, batch, batch_idx):
            mel, mel_mask, pre_ids, pre_mask, post_ids, post_mask, out_ids, out_mask = batch
            embeds, atts, label_ids = self.encode(
                mel, mel_mask, pre_ids, pre_mask, post_ids, post_mask, out_ids, out_mask)
            outputs = self.forward(embeds, atts, label_ids)
            loss = outputs["loss"]
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode only output-portion predictions to avoid input-position noise
            generated_output_text = self._decode_output_only(predicted_ids, label_ids)
            target_text = self.llm_tokenizer.decode(out_ids[0], skip_special_tokens=True)
            
            # Robust field extraction (tolerates malformed JSON from teacher forcing)
            extracted_pred = self._extract_all_fields_robust(generated_output_text)
            extracted_target = self._extract_all_fields_robust(target_text)

            keys = extracted_target.keys()

            for key in keys:
                if key not in extracted_pred:
                    extracted_pred[key] = "NA"

            if 'Transcript' in keys:
                target_transcript = extracted_target['Transcript']
                predicted_transcript = extracted_pred['Transcript']
                wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
                self.log("val/wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Response' in keys:
                target_transcript = extracted_target['Response']
                predicted_transcript = extracted_pred['Response']
                wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
                self.log("val/response_wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'SpeechActivity' in keys:
                target_isspeech = extracted_target['SpeechActivity']
                predicted_isspeech = extracted_pred['SpeechActivity']
                self.log("val/speech_activity", float(target_isspeech.lower()==predicted_isspeech.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Gender' in keys:
                target_gender = extracted_target['Gender']
                predicted_gender = extracted_pred['Gender']
                self.log("val/gender", float(target_gender.lower()==predicted_gender.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Emotion' in keys:
                target_emotion = extracted_target['Emotion']
                predicted_emotion = extracted_pred['Emotion']
                self.log("val/emotion", float(target_emotion.lower()==predicted_emotion.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Age' in keys:
                target_age = extracted_target['Age']
                predicted_age = extracted_pred['Age']
                self.log("val/age", float(target_age.lower()==predicted_age.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Accent' in keys:
                target_accent = extracted_target['Accent']
                predicted_accent = extracted_pred['Accent']
                self.log("val/accent", float(target_accent.lower()==predicted_accent.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return {"val_loss": loss}
    
    def on_validation_epoch_start(self):
        """Select two random validation samples to log for each epoch."""
        self.selected_samples_for_logging = random.sample(range(self.num_validation_samples), 2)

    
    def extract_dictionary(self, input_string):
        pattern = r'<s>\s*(\{.*?\})\s*</s>'
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            dict_string = match.group(1)
            dict_string = re.sub(r',\s*}', '}', dict_string)
            try:
                return json.loads(dict_string)
            except json.JSONDecodeError as e:
                return {}
        else:
            return {}
    
    def extract_prediction_values(self, input_string):
        json_str_match = re.search(r'<s>\s*\{.*?\}\s*</s>', input_string)
        try:
            json_str = json_str_match.group(0)
        except:
            json_str = '{}'
        return self.extract_dictionary(json_str)