"""
finetune_beats.py

BEATs fine-tuning on BSD10k for DCASE 2026 Task 1.

Mirrors the professors' recipe exactly for training:
  - Encoder       : BEATs iter3+ AS2M (pretrained), AudioSet predictor stripped
  - Audio wrapper : ChunkedBEATs(max_audio_seconds=10) -> 768-dim pooled embed
  - Head          : Dropout(0.1) -> Linear(768, 23)
  - Loss          : CrossEntropyLoss
  - Optimizer     : AdamW, lr=3e-5, weight_decay=0.01
  - Schedule      : LambdaLR (linear warmup -> constant -> linear decay)
                    Defaults match README: 1-epoch warmup, decay starts at
                    epoch 1, min_lr=1e-6, max_epochs=10
  - batch=6, gradient_clip=1.0
  - Full FT, no freezing (encoder + head both train)

Three forced corrections vs. their script:
  1. precision=32 (Maxwell GTX/TITAN X cannot do mixed precision; fp16 -> NaN)
  2. Splits via eval_foundation (locked seed=42 20% test, fold 0 by default
     for the val set).
  3. Validation/test scored with the OFFICIAL DCASE metric — verbatim
     hierarchical_prf_weighted from the organizers' evaluate.py, imported via
     a_official_metric.macro_hPRF. Predictions are plain argmax (no
     hierarchy-aware re-weighting, since the organizers do not use it).
     val/hF is what the Bayesian sweep optimizes and what ModelCheckpoint
     monitors.

Run (in tmux):
  CUDA_VISIBLE_DEVICES=GPU-3d29bd5e-738b-b253-4fee-eaeb313a2c6f \
      python ex_dcase_task1_beats_ft_v1.py
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as Faudio
import soundfile as sf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

# Project imports -- both live next to this file
from eval_foundation import (
    CLASSES,
    CLS2IDX,
    IDX2CLS,
    load_metadata,
    get_splits,
)
from a_official_metric import macro_hPRF   # OFFICIAL DCASE metric
from models.beats import BEATs, BEATsConfig, ChunkedBEATs


# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_DATASET_PATH    = Path.home() / "data" / "bsd10k"
DEFAULT_CHECKPOINT_PATH = Path.home() / "checkpoints" / "beats_iter3plus_as2m.pt"
DEFAULT_OUTPUT_ROOT     = Path.home() / "malach26dcasetask1" / "outputs" / "a_beats_ft_official"
DEFAULT_WANDB_PROJECT   = "dcase2026-task1"
SAMPLE_RATE             = 16000  # BEATs requirement


# ─── Dataset (loads raw .wav, resamples to 16 kHz mono on the fly) ───────────
class BSD10kWaveformDataset(Dataset):
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
        waveform, sr = sf.read(str(wav_path), always_2d=False, dtype="float32")
        waveform = np.asarray(waveform, dtype=np.float32)

        # mono
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        # resample to 16 kHz if needed
        if sr != SAMPLE_RATE:
            tensor = torch.from_numpy(waveform).unsqueeze(0)
            tensor = Faudio.resample(tensor, sr, SAMPLE_RATE)
            waveform = tensor.squeeze(0).numpy().astype(np.float32, copy=False)

        return {
            "waveform": waveform,
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


# ─── LR schedule (matches professors' build_lr_lambda exactly) ───────────────
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


# ─── Metric: the OFFICIAL DCASE metric on plain argmax predictions ────────────
def compute_official_hPRF(logits_np, labels_np):
    """Returns (hP, hR, hF) under the organizers' official metric (verbatim from
    evaluate.py, see a_official_metric.py). Predictions are plain argmax — the
    organizers do NOT apply hierarchy-aware re-weighting."""
    preds = logits_np.argmax(axis=-1)
    y_true_str = [IDX2CLS[int(t)] for t in labels_np]
    y_pred_str = [IDX2CLS[int(p)] for p in preds]
    return macro_hPRF(y_true_str, y_pred_str, lambda_param=0.75)


# ─── Lightning module ────────────────────────────────────────────────────────
class BEATsClassifier(pl.LightningModule):
    def __init__(
        self,
        beats_model,
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
        freeze_encoder=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["beats_model"])

        self.beats = ChunkedBEATs(
            beats_model,
            sample_rate=sample_rate,
            max_audio_seconds=max_audio_seconds,
        )
        if freeze_encoder:
            for p in self.beats.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(encoder_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self._val_logits, self._val_labels = [], []
        self._test_logits, self._test_labels = [], []

    def forward(self, waveforms, padding_mask):
        pooled = self.beats(waveforms, padding_mask)
        return self.classifier(self.dropout(pooled))

    def training_step(self, batch, batch_idx):
        logits = self(batch["waveforms"], batch["padding_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        acc = (logits.argmax(-1) == batch["labels"]).float().mean()
        bs = batch["labels"].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train/accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["waveforms"], batch["padding_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch["labels"].size(0))
        self._val_logits.append(logits.detach().cpu().numpy())
        self._val_labels.append(batch["labels"].detach().cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        logits_np = np.concatenate(self._val_logits, axis=0)
        labels_np = np.concatenate(self._val_labels, axis=0)

        hP, hR, hF = compute_official_hPRF(logits_np, labels_np)

        # OFFICIAL DCASE metric only — the wrong rulers (hf_correct / hf_professors)
        # have been removed. val/hF is the leaderboard metric and the sweep target.
        self.log("val/hP", float(hP), on_step=False, on_epoch=True)
        self.log("val/hR", float(hR), on_step=False, on_epoch=True)
        self.log("val/hF", float(hF), on_step=False, on_epoch=True, prog_bar=True)

        self._val_logits.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["waveforms"], batch["padding_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("test/loss", loss, on_step=False, on_epoch=True,
                 batch_size=batch["labels"].size(0))
        self._test_logits.append(logits.detach().cpu().numpy())
        self._test_labels.append(batch["labels"].detach().cpu().numpy())
        return loss

    def on_test_epoch_end(self):
        if not self._test_logits:
            return
        logits_np = np.concatenate(self._test_logits, axis=0)
        labels_np = np.concatenate(self._test_labels, axis=0)

        hP, hR, hF = compute_official_hPRF(logits_np, labels_np)
        acc = float((logits_np.argmax(-1) == labels_np).mean())

        self.log("test/hP", float(hP))
        self.log("test/hR", float(hR))
        self.log("test/hF", float(hF))
        self.log("test/accuracy", float(acc))

        print()
        print("=" * 62)
        print("LOCKED TEST SET RESULTS  (seed=42, 20 % holdout, 2192 samples)")
        print("Scored with the OFFICIAL DCASE metric (evaluate.py verbatim)")
        print("=" * 62)
        print(f"  Plain accuracy                : {acc:.4f}")
        print(f"  hierarchical precision (hP)   : {hP:.4f}")
        print(f"  hierarchical recall    (hR)   : {hR:.4f}")
        print(f"  hierarchical f1        (hF)   : {hF:.4f}   <- leaderboard")
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
    p.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--lr-decay-start-epoch", type=float, default=1.0)
    p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--gradient-clip-val", type=float, default=1.0)
    p.add_argument("--max-audio-seconds", type=float, default=10.0)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="1")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    # 1. Data + splits via eval_foundation
    dataset_path = Path(args.dataset_path).expanduser()
    df = load_metadata(dataset_path)
    test_idx, folds = get_splits(df, seed=args.seed)
    if not (0 <= args.fold < len(folds)):
        raise ValueError(f"fold {args.fold} not in [0, {len(folds) - 1}]")
    train_idx, val_idx = folds[args.fold]
    print(f"\nFold {args.fold}:  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)} (locked)")

    train_ds = BSD10kWaveformDataset(df, train_idx, dataset_path)
    val_ds   = BSD10kWaveformDataset(df, val_idx,   dataset_path)
    test_ds  = BSD10kWaveformDataset(df, test_idx,  dataset_path)

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

    # 2. Build BEATs encoder from checkpoint; strip AudioSet predictor
    print(f"Loading BEATs checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(
        str(Path(args.checkpoint_path).expanduser()),
        map_location="cpu",
        weights_only=False,
    )
    cfg = BEATsConfig(ckpt["cfg"])
    cfg.finetuned_model = False
    beats_model = BEATs(cfg)
    state_dict = {k: v for k, v in ckpt["model"].items() if not k.startswith("predictor.")}
    missing, unexpected = beats_model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    non_pred_missing = [k for k in missing if not k.startswith("predictor")]
    if non_pred_missing:
        raise RuntimeError(f"Missing encoder keys: {non_pred_missing}")
    encoder_dim = cfg.encoder_embed_dim  # 768

    # 3. Schedule sizing
    train_batches_per_epoch = max(1, math.ceil(len(train_ds) / args.batch_size))
    update_steps_per_epoch  = train_batches_per_epoch
    total_steps             = max(1, update_steps_per_epoch * args.max_epochs)
    warmup_steps            = max(0, int(args.warmup_epochs * update_steps_per_epoch))
    decay_start_step        = (
        int(args.lr_decay_start_epoch * update_steps_per_epoch)
        if args.lr_decay_start_epoch is not None else None
    )
    print(f"Schedule: warmup_steps={warmup_steps}  decay_start_step={decay_start_step}  total_steps={total_steps}")

    # 4. Lightning module
    model = BEATsClassifier(
        beats_model=beats_model,
        encoder_dim=encoder_dim,
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
        freeze_encoder=args.freeze_encoder,
    )

    # 5. Run dir + W&B
    run_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_a_beats_ft_official_fold{args.fold}"
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
        monitor="val/hF",               # OFFICIAL DCASE metric (sweep target)
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
        precision=32,                          # FORCED for Maxwell
        gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger,
        callbacks=[ckpt_cb, lr_cb],
        default_root_dir=str(run_dir),
        deterministic=False,                   # ChunkedBEATs index_add not deterministic
        log_every_n_steps=10,
    )

    # 8. Train, then evaluate the best checkpoint on the LOCKED test set
    trainer.fit(model, train_loader, val_loader)
    print(f"\nBest val/hF = {ckpt_cb.best_model_score}  -- {ckpt_cb.best_model_path}")
    trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()
