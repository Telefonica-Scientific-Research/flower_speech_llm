"""PyTorch Lightning module for Voxtral federated speech-LLM training.

Unlike the WavLM+connector+LLM pipeline in trainer.py, Voxtral is a single
end-to-end model.  The collator (dataset_voxtral.py) builds the full
input_ids / attention_mask / labels batch; this module just does the forward
pass and logs metrics.
"""

import torch
import pytorch_lightning as pl
from jiwer import wer as compute_wer


class VoxtralLightning(pl.LightningModule):
    """Lightning wrapper for VoxtralForConditionalGeneration with LoRA."""

    def __init__(self, model, processor, max_lr=5e-5, warmup_steps=100,
                 total_training_step=10_000_000, **kwargs):
        super().__init__()
        # Don't call save_hyperparameters() — model/processor are not serializable
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer

        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_training_step = total_training_step

    # ---- expose tokenizer for compatibility with build_loaders ----
    @property
    def llm_tokenizer(self):
        return self.tokenizer

    def configure_optimizers(self):
        # Separate LR for connector (higher) vs LoRA params (lower)
        connector_params = []
        lora_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "multi_modal_projector" in name:
                connector_params.append(p)
            else:
                lora_params.append(p)

        param_groups = []
        if connector_params:
            param_groups.append({"params": connector_params, "lr": self.max_lr})
        if lora_params:
            param_groups.append({"params": lora_params, "lr": self.max_lr})

        return torch.optim.AdamW(param_groups, lr=self.max_lr, weight_decay=0.01)

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        # batch is a dict from VoxtralCollator: input_ids, attention_mask, labels, + audio keys
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val/loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Compute WER on greedy argmax (no beam search in validation)
        logits = outputs.logits
        labels = batch["labels"]
        predicted_ids = torch.argmax(logits, dim=-1).cpu()
        del outputs, logits  # free GPU memory

        # Decode only the label portion (where labels != -100)
        labels = labels.cpu()
        for i in range(labels.size(0)):
            label_mask = labels[i] != -100
            target_ids = labels[i][label_mask]
            # Align predicted to same positions
            pred_ids = predicted_ids[i][label_mask]

            target_text = self.tokenizer.decode(target_ids, skip_special_tokens=True).strip()
            pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

            if target_text:
                wer_val = compute_wer(target_text.lower(), pred_text.lower())
                self.log("val/wer", wer_val, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
