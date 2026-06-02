"""
ex_dcase_task1_clap_ft_v2.py

CLAP fine-tuning on BSD10k for DCASE 2026 Task 1.

Same training recipe and audio pipeline as v1. The ONLY changes in v2 are about
clarity of monitoring — the model, optimizer, schedule, and data are untouched:

  CHANGES vs v1:
  1. W&B now logs ONLY the metrics that matter:
        train/loss        (epoch)   - is the model learning?
        val/loss          (epoch)   - overfitting check
        val/hf_correct    (epoch)   - THE metric (monitored for checkpointing)
        val/hf_professors (epoch)   - for comparison to the professors' number
     The val/hf_correct_precision and _recall sub-metrics from v1 are removed.
     Those shared a string prefix with val/hf_correct, which prevented the bare
     val/hf_correct from showing up in W&B. Removing them fixes that AND cuts
     noise. Precision/recall are still printed in the final test block.
  2. A clear CONFIG block prints at startup listing every hyperparameter, so any
     run (or sweep trial) makes its settings obvious at a glance.
  3. NaN/inf guard on the logged scores, just in case.

Recipe (unchanged from v1):
  - Encoder      : LAION-CLAP audio branch, pretrained on LAION-Audio-630k
  - Audio wrapper: ChunkedCLAP(10s) -> mean-pool segments
  - Head         : Dropout(0.1) -> Linear(512, 23)
  - Loss         : CrossEntropyLoss
  - Optimizer    : AdamW, lr=1e-5 default, weight_decay=0.01
  - Schedule     : warmup -> linear decay to min_lr
  - Precision    : 32 (Maxwell), gradient_clip=1.0, full FT
  - Audio        : 48 kHz mono, int16 round-trip, 10s chunks
  - Splits+metric: eval_foundation (seed=42 locked test)

All six sweepable hyperparameters are CLI args (learning-rate, weight-decay,
warmup-epochs, lr-decay-start-epoch, min-learning-rate, max-epochs), so this
script drops directly into a W&B Bayesian sweep with no code changes.

Run (in tmux):
  CUDA_VISIBLE_DEVICES=<gpu-uuid> python ex_dcase_task1_clap_ft_v2.py --batch-size 4
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import warnings
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Quiet down library logs
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
for name in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(name).setLevel(logging.WARNING)

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as Faudio
import soundfile as sf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

import laion_clap

from eval_foundation import (
    CLASSES,
    CLS2IDX,
    IDX2CLS,
    load_metadata,
    get_splits,
    evaluate_probs,
)
from models.audio_wrappers import ArbitraryLengthAudioWrapper, mean_segment_outputs


# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_DATASET_PATH    = Path.home() / "data" / "bsd10k"
DEFAULT_OUTPUT_ROOT     = Path.home() / "malach26dcasetask1" / "outputs" / "clap_ft"
DEFAULT_WANDB_PROJECT   = "dcase2026-task1"
SAMPLE_RATE             = 48000  # CLAP's native sample rate
CLAP_EMBED_DIM          = 512    # CLAP audio embedding dimension


# ─── Audio preprocessing ─────────────────────────────────────────────────────
def int16_round_trip(wav: np.ndarray) -> np.ndarray:
    """Quantize to int16 and back (per LAION-CLAP official example)."""
    wav = np.clip(wav, -1.0, 1.0)
    wav_int16 = (wav * 32767.0).astype(np.int16)
    return (wav_int16.astype(np.float32) / 32767.0)


class BSD10kWaveformDatasetCLAP(Dataset):
    """Loads raw waveforms at 48 kHz for CLAP."""

    def __init__(self, df, indices, dataset_path):
        self.df = df.reset_index(drop=True)
        self.indices = [int(i) for i in indices]
        self.audio_dir = Path(dataset_path).expanduser() / "audio"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.df.iloc[self.indices[i]]
        sound_id = int(row["sound_id"])
        label_str = str(row["class"])

        wav_path = self.audio_dir / f"{sound_id}.wav"
        wav, sr = sf.read(str(wav_path), always_2d=False, dtype="float32")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        # int16 quantization round-trip (per LAION-CLAP official example)
        wav = int16_round_trip(wav)

        # resample to 48 kHz if needed
        if sr != SAMPLE_RATE:
            tensor = torch.from_numpy(wav).unsqueeze(0)
            tensor = Faudio.resample(tensor, sr, SAMPLE_RATE)
            wav = tensor.squeeze(0).numpy().astype(np.float32, copy=False)

        return {
            "waveform": wav,
            "label": CLS2IDX[label_str],
            "sound_id": sound_id,
        }


def collate_waveforms(batch):
    """Pad to longest waveform in the batch. padding_mask: True == padded."""
    lengths = [len(item["waveform"]) for item in batch]
    max_len = max(lengths)
    bsz = len(batch)
    waveforms = torch.zeros((bsz, max_len), dtype=torch.float32)
    padding_mask = torch.ones((bsz, max_len), dtype=torch.bool)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        n = len(item["waveform"])
        waveforms[i, :n] = torch.from_numpy(item["waveform"])
        padding_mask[i, :n] = False
    return {"waveforms": waveforms, "padding_mask": padding_mask, "labels": labels}


# ─── ChunkedCLAP wrapper (mirrors ChunkedBEATs) ──────────────────────────────
def _extract_clap_embeddings(clap_module, waveforms, padding_mask=None):
    """Forward a batch of fixed-length 10s waveforms through CLAP's audio branch.
    Returns [N, 512] audio embeddings."""
    return clap_module.get_audio_embedding_from_data(x=waveforms, use_tensor=True)


class ChunkedCLAP(ArbitraryLengthAudioWrapper):
    """Chunk any-length audio into 10s segments, encode each, mean-pool.
    Output shape: [B, 512] regardless of input length."""

    def __init__(self, clap_module, sample_rate=SAMPLE_RATE, max_audio_seconds=10.0):
        super().__init__(
            clap_module,
            sample_rate=sample_rate,
            max_audio_seconds=max_audio_seconds,
            segment_forward=_extract_clap_embeddings,
            aggregate_outputs=mean_segment_outputs,
        )


# ─── LR schedule (unchanged from v1) ─────────────────────────────────────────
def build_lr_lambda(warmup_steps, decay_start_step, total_steps, min_lr_scale):
    warmup_steps = max(0, warmup_steps)
    total_steps = max(1, total_steps)
    min_lr_scale = min(max(0.0, min_lr_scale), 1.0)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            if warmup_steps == 1:
                return 0.0
            return float(step) / float(warmup_steps - 1)
        if decay_start_step is None or decay_start_step >= total_steps:
            return 1.0
        if step < decay_start_step:
            return 1.0
        span = total_steps - decay_start_step
        if span <= 1:
            return min_lr_scale
        progress = min(step - decay_start_step, span - 1)
        frac = float(progress) / float(span - 1)
        return 1.0 - ((1.0 - min_lr_scale) * frac)

    return lr_lambda


# ─── Metrics (unchanged computation; cleaner logging) ────────────────────────
def _safe(x):
    """Guard against NaN/inf so W&B always receives a clean float."""
    x = float(x)
    return x if math.isfinite(x) else 0.0


def compute_hf_correct(logits_np, labels_np):
    """Returns (precision, recall, f_score) via eval_foundation."""
    probs = torch.softmax(torch.from_numpy(logits_np), dim=-1)
    return evaluate_probs(probs, labels_np, use_hierarchy=True)


def compute_hf_professors(logits_np, labels_np):
    """Reproduces the professors' buggy partial_match on raw argmax."""
    preds = logits_np.argmax(axis=-1)

    def partial_match(y_t, y_p, d=0.75):
        if y_t == y_p:
            return 1
        if y_t.split('-')[0] == y_t.split('-')[0]:  # bug from professors' code
            return d / 2
        return 0

    y_true_str = [IDX2CLS[int(t)] for t in labels_np]
    y_pred_str = [IDX2CLS[int(p)] for p in preds]

    hP, hR, hF = {}, {}, {}
    for c in set(y_true_str):
        prec_terms = [partial_match(yt, yp) for yt, yp in zip(y_true_str, y_pred_str) if yp == c]
        rec_terms  = [partial_match(yt, yp) for yt, yp in zip(y_true_str, y_pred_str) if yt == c]
        hP[c] = float(np.mean(prec_terms)) if prec_terms else 0.0
        hR[c] = float(np.mean(rec_terms))  if rec_terms  else 0.0
        denom = hP[c] + hR[c]
        hF[c] = (2 * hP[c] * hR[c] / denom) if denom > 0 else 0.0

    return (
        float(np.mean(list(hP.values()))),
        float(np.mean(list(hR.values()))),
        float(np.mean(list(hF.values()))),
    )


# ─── Lightning module ────────────────────────────────────────────────────────
class CLAPClassifier(pl.LightningModule):
    def __init__(
        self,
        clap_module,
        encoder_dim,
        num_classes,
        head_dropout,
        max_audio_seconds,
        sample_rate,
        learning_rate,
        weight_decay,
        warmup_steps,
        decay_start_step,
        total_steps,
        min_learning_rate,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["clap_module"])

        self.clap = ChunkedCLAP(
            clap_module,
            sample_rate=sample_rate,
            max_audio_seconds=max_audio_seconds,
        )
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(encoder_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self._val_logits, self._val_labels = [], []
        self._test_logits, self._test_labels = [], []

    def forward(self, waveforms, padding_mask):
        pooled = self.clap(waveforms, padding_mask)
        return self.classifier(self.dropout(pooled))

    def training_step(self, batch, batch_idx):
        logits = self(batch["waveforms"], batch["padding_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        # ONLY epoch-level train/loss is logged — no step-level noise, no accuracy
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch["labels"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["waveforms"], batch["padding_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("val/loss", loss, on_step=False, on_epoch=True,
                 batch_size=batch["labels"].size(0))
        self._val_logits.append(logits.detach().cpu().numpy())
        self._val_labels.append(batch["labels"].detach().cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        logits_np = np.concatenate(self._val_logits, axis=0)
        labels_np = np.concatenate(self._val_labels, axis=0)

        _, _, hf_c = compute_hf_correct(logits_np, labels_np)
        _, _, hf_p = compute_hf_professors(logits_np, labels_np)

        # Log ONLY the two F-scores. No _precision/_recall sub-metrics, so
        # val/hf_correct no longer shares a prefix and shows up cleanly in W&B.
        self.log("val/hf_correct", _safe(hf_c), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/hf_professors", _safe(hf_p), on_step=False, on_epoch=True)

        self._val_logits.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["waveforms"], batch["padding_mask"])
        self._test_logits.append(logits.detach().cpu().numpy())
        self._test_labels.append(batch["labels"].detach().cpu().numpy())

    def on_test_epoch_end(self):
        if not self._test_logits:
            return
        logits_np = np.concatenate(self._test_logits, axis=0)
        labels_np = np.concatenate(self._test_labels, axis=0)

        hp_c, hr_c, hf_c = compute_hf_correct(logits_np, labels_np)
        _, _, hf_p = compute_hf_professors(logits_np, labels_np)
        acc = float((logits_np.argmax(-1) == labels_np).mean())

        # Log only the headline F-scores + accuracy (no prefix-colliding submetrics)
        self.log("test/hf_correct", _safe(hf_c))
        self.log("test/hf_professors", _safe(hf_p))
        self.log("test/accuracy", _safe(acc))

        # Precision/recall are printed here (not logged to W&B) for the record.
        print()
        print("=" * 62)
        print("LOCKED TEST SET RESULTS  (seed=42, 20 % holdout, 2192 samples)")
        print("=" * 62)
        print(f"  Plain accuracy                         : {acc:.4f}")
        print(f"  hf_correct     (vs 0.8226 ensemble)    : {hf_c:.4f}")
        print(f"      precision                          : {hp_c:.4f}")
        print(f"      recall                             : {hr_c:.4f}")
        print(f"  hf_professors  (vs professors' ~0.82)  : {hf_p:.4f}")
        print("=" * 62)

        self._test_logits.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.warmup_steps <= 0 and self.hparams.decay_start_step is None:
            return opt
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=build_lr_lambda(
                warmup_steps=self.hparams.warmup_steps,
                decay_start_step=self.hparams.decay_start_step,
                total_steps=self.hparams.total_steps,
                min_lr_scale=(
                    self.hparams.min_learning_rate / self.hparams.learning_rate
                    if self.hparams.learning_rate > 0 else 0.0
                ),
            ),
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }


# ─── CLI + main ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)   # Maxwell-safe default
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--lr-decay-start-epoch", type=float, default=1.0)
    p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--gradient-clip-val", type=float, default=1.0)
    p.add_argument("--max-audio-seconds", type=float, default=10.0)
    p.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="1")
    return p.parse_args()


def print_config(args):
    """Print every hyperparameter clearly at startup."""
    print("=" * 62)
    print("CLAP FINE-TUNING — run configuration")
    print("=" * 62)
    print(f"  learning_rate         : {args.learning_rate:g}")
    print(f"  weight_decay          : {args.weight_decay:g}")
    print(f"  warmup_epochs         : {args.warmup_epochs:g}")
    print(f"  lr_decay_start_epoch  : {args.lr_decay_start_epoch:g}")
    print(f"  min_learning_rate     : {args.min_learning_rate:g}")
    print(f"  max_epochs            : {args.max_epochs}")
    print(f"  batch_size            : {args.batch_size}")
    print(f"  head_dropout          : {args.head_dropout:g}")
    print(f"  gradient_clip_val     : {args.gradient_clip_val:g}")
    print(f"  seed                  : {args.seed}")
    print(f"  fold                  : {args.fold}")
    print("=" * 62)


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    print_config(args)

    # 1. Data + splits via eval_foundation
    dataset_path = Path(args.dataset_path).expanduser()
    df = load_metadata(dataset_path)
    test_idx, folds = get_splits(df, seed=args.seed)
    if not (0 <= args.fold < len(folds)):
        raise ValueError(f"fold {args.fold} not in [0, {len(folds) - 1}]")
    train_idx, val_idx = folds[args.fold]
    print(f"Fold {args.fold}:  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)} (locked)")

    train_ds = BSD10kWaveformDatasetCLAP(df, train_idx, dataset_path)
    val_ds   = BSD10kWaveformDatasetCLAP(df, val_idx,   dataset_path)
    test_ds  = BSD10kWaveformDatasetCLAP(df, test_idx,  dataset_path)

    common_dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_waveforms,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **common_dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common_dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common_dl_kwargs)

    # 2. Build CLAP module (downloads default checkpoint on first run)
    print("Loading LAION-CLAP (downloads ~600MB checkpoint to ~/.cache/laion_clap/ on first run)...")
    clap_module = laion_clap.CLAP_Module(enable_fusion=False)
    clap_module.load_ckpt()
    print(f"CLAP loaded. Audio embedding dim = {CLAP_EMBED_DIM}")

    # 3. Schedule sizing
    update_steps_per_epoch = max(1, math.ceil(len(train_ds) / args.batch_size))
    total_steps = max(1, update_steps_per_epoch * args.max_epochs)
    warmup_steps = max(0, int(args.warmup_epochs * update_steps_per_epoch))
    decay_start_step = (
        int(args.lr_decay_start_epoch * update_steps_per_epoch)
        if args.lr_decay_start_epoch is not None else None
    )
    print(f"Schedule: warmup_steps={warmup_steps}  decay_start_step={decay_start_step}  total_steps={total_steps}")

    # 4. Lightning module
    model = CLAPClassifier(
        clap_module=clap_module,
        encoder_dim=CLAP_EMBED_DIM,
        num_classes=len(CLASSES),
        head_dropout=args.head_dropout,
        max_audio_seconds=args.max_audio_seconds,
        sample_rate=SAMPLE_RATE,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        decay_start_step=decay_start_step,
        total_steps=total_steps,
        min_learning_rate=args.min_learning_rate,
    )

    # 5. Run dir + W&B
    run_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_clap_ft_fold{args.fold}"
        f"_lr{args.learning_rate:g}"
        f"_{uuid4().hex[:6]}"
    )
    run_dir = Path(args.output_root).expanduser() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    wandb_logger = None
    if args.wandb_mode != "disabled":
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=run_id,
            save_dir=str(run_dir),
            mode=args.wandb_mode,
        )
        wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

    # 6. Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best-epoch{epoch:02d}",
        monitor="val/hf_correct",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # 7. Trainer
    devices = int(args.devices) if args.devices.isdigit() else args.devices
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=devices,
        max_epochs=args.max_epochs,
        precision=32,
        gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger,
        callbacks=[ckpt_cb, lr_cb],
        default_root_dir=str(run_dir),
        deterministic=False,
        log_every_n_steps=10,
    )

    # 8. Train, then evaluate the best checkpoint on the locked test set
    trainer.fit(model, train_loader, val_loader)
    print(f"\nBest val/hf_correct = {ckpt_cb.best_model_score}  -- {ckpt_cb.best_model_path}")
    trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()
