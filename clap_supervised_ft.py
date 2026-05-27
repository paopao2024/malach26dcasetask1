"""
CLAP supervised fine-tune — OFFICIAL audio-only recipe (per LAION-CLAP README
architecture diagram, "Supervised Audio Classification" panel).

This is the recipe we had NOT run before. v8-v11 all kept the frozen CLAP TEXT
embedding fused in alongside the fine-tuned audio encoder — a hybrid. The
official diagram shows supervised fine-tuning as:

    Audio -> Audio Encoder -> Projection Layers -> Class Prob. Vector (1,C)
    (the "Finetune" brace covers BOTH the audio encoder and the projection layers)
    NO TEXT ENCODER in this path.

So this script:
  - drops text entirely (audio-only),
  - applies the int16 quantization the README's example uses BEFORE encoding
    (int16_to_float32(float32_to_int16(x))) — we skipped this before,
  - feeds 48 kHz audio through CLAP's audio branch,
  - attaches trainable PROJECTION LAYERS -> 23 classes,
  - fine-tunes encoder + projection together.

Defaults are conservative; override via CLI once the professors give their
exact projection-head / LR settings. Same locked-test eval as every stage
(eval_foundation), comparable to the 0.8226 frozen ensemble.

NOTE ON COST: this runs raw audio through CLAP every step (like v8-v11), so it
is the SLOW path on the Maxwell GPU, not the fast frozen-feature path. Use the
fold-1 decision gate before committing to all 5 folds. RUN IN TMUX.
"""

import os
import warnings
import logging
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse

import soundfile as sf
import torchaudio
import laion_clap

from eval_foundation import (
    CLASSES, CLS2IDX, IDX2CLS,
    load_metadata, get_splits, evaluate_probs, SEED,
)

CLAP_SR = 48000
NATIVE_SR = 44100
MAX_SAMPLES = CLAP_SR * 10   # 10 s windows (matches CLAP's design)


# README quantization helpers — applied before encoding, as in the official example
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype(np.int16)


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma; self.ls = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class AudioDataset(Dataset):
    """Audio-only. Returns the quantized 48 kHz waveform + label (NO text)."""
    def __init__(self, df, dataset_path, augment=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(dataset_path) / "audio"
        self.resampler = torchaudio.transforms.Resample(NATIVE_SR, CLAP_SR)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load(self, sid):
        wav, sr = sf.read(self.audio_dir / f"{sid}.wav", dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # int16 quantization, exactly as the README example does before encoding
        wav = int16_to_float32(float32_to_int16(wav))
        wav = torch.from_numpy(wav)
        if sr != CLAP_SR:
            wav = self.resampler(wav) if sr == NATIVE_SR else \
                  torchaudio.transforms.Resample(sr, CLAP_SR)(wav)
        if wav.numel() >= MAX_SAMPLES:
            wav = wav[:MAX_SAMPLES]
        else:
            wav = F.pad(wav, (0, MAX_SAMPLES - wav.numel()))
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = self._load(str(row.sound_id))
        if self.augment and torch.rand(1).item() < 0.5:
            wav = wav + 0.005 * torch.randn_like(wav)
        return wav, torch.tensor(CLS2IDX[row["class"]], dtype=torch.long)


class ProjectionHead(nn.Module):
    """The 'Projection Layers' from the diagram: 512 -> hidden -> 23."""
    def __init__(self, in_dim=512, hidden=256, n_classes=23, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class SupervisedClap(pl.LightningModule):
    def __init__(self, lr_proj=1e-3, lr_encoder=1e-5, gamma=2.0,
                 label_smoothing=0.1, warmup_epochs=2, proj_hidden=256):
        super().__init__()
        self.save_hyperparameters()
        self.lr_proj = lr_proj
        self.lr_encoder = lr_encoder
        self.warmup_epochs = warmup_epochs
        self.loss_fn = FocalLossWithSmoothing(gamma=gamma, label_smoothing=label_smoothing)
        self.val_outputs = []

        self.clap = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap.load_ckpt()
        for p in self.clap.parameters():
            p.requires_grad = False
        self.audio_branch = self.clap.model.audio_branch
        self.head = ProjectionHead(in_dim=512, hidden=proj_hidden)

        self._encoder_unfrozen = False
        n = sum(p.numel() for p in self.audio_branch.parameters())
        print(f"  audio branch {n/1e6:.1f}M params (frozen for {warmup_epochs} warmup epochs); "
              f"projection head trains from start; AUDIO-ONLY, no text")

    def _unfreeze_encoder(self):
        for p in self.audio_branch.parameters():
            p.requires_grad = True
        # freeze only BatchNorm (small-batch stability); LayerNorms train
        nbn = 0
        for m in self.audio_branch.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                nbn += 1
        self._encoder_unfrozen = True
        print(f"  >> warmup done: encoder unfrozen ({nbn} BatchNorm frozen)")

    def encode(self, wav):
        return self.clap.get_audio_embedding_from_data(x=wav, use_tensor=True)  # [B,512]

    def forward(self, wav):
        return self.head(self.encode(wav))

    def configure_optimizers(self):
        opt = torch.optim.Adam([
            {"params": self.head.parameters(),         "lr": self.lr_proj},
            {"params": self.audio_branch.parameters(), "lr": self.lr_encoder},
        ], weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [sch]

    def on_train_epoch_start(self):
        if (not self._encoder_unfrozen) and self.current_epoch >= self.warmup_epochs:
            self._unfreeze_encoder()
        if self._encoder_unfrozen:
            for m in self.audio_branch.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

    def training_step(self, batch, _):
        wav, y = batch
        loss = self.loss_fn(self(wav), y)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        wav, y = batch
        logits = self(wav)
        self.val_outputs.append({"logits": logits.detach().cpu(), "y": y.cpu()})

    def on_validation_epoch_end(self):
        probs = F.softmax(torch.cat([o["logits"] for o in self.val_outputs]), dim=1)
        y = torch.cat([o["y"] for o in self.val_outputs])
        _, _, hf = evaluate_probs(probs, y.tolist(), use_hierarchy=True)
        self.log_dict({"val/hF_hier": hf})
        self.val_outputs.clear()


@torch.no_grad()
def test_probs(model, df, dataset_path, device, batch_size=16):
    dl = DataLoader(AudioDataset(df, dataset_path), batch_size=batch_size,
                    shuffle=False, num_workers=4)
    out = [F.softmax(model(w.to(device)), dim=1).cpu() for w, _ in dl]
    return torch.cat(out)


def run(dataset_path, n_epochs=20, batch_size=8, accum_steps=4, patience=6,
        gamma=2.0, label_smoothing=0.1, lr_proj=1e-3, lr_encoder=1e-5,
        warmup_epochs=2, proj_hidden=256, only_fold=None,
        ckpt_dir="checkpoints_clap_sup"):
    pl.seed_everything(SEED, workers=True)
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_metadata(dataset_path)
    test_idx, folds = get_splits(df, seed=SEED)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_labels = [CLS2IDX[c] for c in test_df["class"]]
    print(f"AUDIO-ONLY supervised CLAP fine-tune (official recipe)")
    print(f"  lr_proj={lr_proj}, lr_encoder={lr_encoder}, warmup={warmup_epochs}, "
          f"proj_hidden={proj_hidden}, batch={batch_size}x{accum_steps}accum, fp32")
    if only_fold is not None:
        print(f"  >> FOLD {only_fold} ONLY (decision gate)")

    test_probs_list, cv = [], []
    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        if only_fold is not None and fold != only_fold:
            continue
        print(f"\n── fold {fold}/5 ──")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df   = df.iloc[va_idx].reset_index(drop=True)
        train_dl = DataLoader(AudioDataset(train_df, dataset_path, augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dl   = DataLoader(AudioDataset(val_df, dataset_path),
                              batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        logger = WandbLogger(project="dcase2026-task1", name=f"clap_sup_fold{fold}")
        cbs = [
            EarlyStopping(monitor="val/hF_hier", patience=patience, mode="max"),
            ModelCheckpoint(dirpath=str(ckpt_dir), filename=f"clap_sup_fold{fold}",
                            monitor="val/hF_hier", mode="max", save_top_k=1,
                            save_weights_only=True, enable_version_counter=False),
        ]
        trainer = pl.Trainer(max_epochs=n_epochs, min_epochs=warmup_epochs + 6,
                             logger=logger, accelerator="auto", devices=1,
                             precision="32-true", gradient_clip_val=1.0,
                             accumulate_grad_batches=accum_steps,
                             enable_progress_bar=False, enable_model_summary=False,
                             callbacks=cbs, log_every_n_steps=20)
        model = SupervisedClap(lr_proj=lr_proj, lr_encoder=lr_encoder, gamma=gamma,
                               label_smoothing=label_smoothing, warmup_epochs=warmup_epochs,
                               proj_hidden=proj_hidden)
        trainer.fit(model, train_dl, val_dl)
        best = SupervisedClap.load_from_checkpoint(cbs[1].best_model_path,
                                                   map_location=device).eval().to(device)
        tp = test_probs(best, test_df, dataset_path, device, batch_size=batch_size)
        test_probs_list.append(tp)
        _, _, hf = evaluate_probs(tp, test_labels, use_hierarchy=True)
        cv.append(hf)
        print(f"  fold {fold} on locked test: {hf:.4f}")
        logger.experiment.finish()
        del model, trainer
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"CLAP SUPERVISED (audio-only, official recipe) — locked test")
    print(f"{'='*60}")
    for i, s in enumerate(cv, 1):
        print(f"  fold {i}: {s:.4f}")
    if len(cv) > 1:
        ens = torch.stack(test_probs_list).mean(dim=0)
        _, _, ens_hf = evaluate_probs(ens, test_labels, use_hierarchy=True)
        print(f"  mean single: {float(np.mean(cv)):.4f}")
        print(f"  ensemble:    {ens_hf:.4f}")
    print(f"{'-'*60}")
    print(f"  frozen ensemble reference: 0.8226")
    print(f"  frozen single v6:          0.8048")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--lr_proj", type=float, default=1e-3)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--proj_hidden", type=int, default=256)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints_clap_sup")
    a = p.parse_args()
    run(a.dataset_path, n_epochs=a.epochs, batch_size=a.batch_size, accum_steps=a.accum_steps,
        patience=a.patience, gamma=a.gamma, label_smoothing=a.label_smoothing,
        lr_proj=a.lr_proj, lr_encoder=a.lr_encoder, warmup_epochs=a.warmup_epochs,
        proj_hidden=a.proj_hidden, only_fold=a.fold, ckpt_dir=a.ckpt_dir)
