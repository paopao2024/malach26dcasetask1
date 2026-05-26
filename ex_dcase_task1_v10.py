"""
DCASE 2026 Task 1 - v10 (= v9 config fix: BN frozen + encoder_lr 3e-6)
Fine-tune CLAP's audio encoder (HTSAT) end-to-end, fused with frozen
precomputed CLAP text embeddings.

Difference from v6:
- v6 used a PRECOMPUTED 512-d CLAP audio embedding (frozen .npy).
- v8 loads the RAW .wav, runs it live through CLAP's HTSAT audio encoder,
  and BACKPROPS through the top of that encoder. The text half stays
  frozen (precomputed .npy), exactly as in v6.

Unfreezing policy (conservative first run):
- Freeze the entire CLAP model.
- Unfreeze ONLY the last 2 HTSAT transformer blocks + the audio projection.
- Train those + the MLP head.

Everything else (focal loss, label smoothing, hierarchy-aware eval,
5-fold CV, seed 42) is identical to v6 so numbers are directly comparable.
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
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from pathlib import Path
import argparse

import soundfile as sf
import torchaudio
import laion_clap


# ──────────────────────────────────────────────────────────────────────
# Taxonomy (identical to v6)
# ──────────────────────────────────────────────────────────────────────
BST = {
    "m-sp": "m",  "m-si": "m",  "m-m": "m",
    "is-p": "is", "is-s": "is", "is-w": "is", "is-k": "is", "is-e": "is",
    "sp-s": "sp", "sp-c": "sp", "sp-p": "sp",
    "fx-o": "fx", "fx-v": "fx", "fx-m": "fx", "fx-h": "fx",
    "fx-a": "fx", "fx-n": "fx", "fx-ex": "fx", "fx-el": "fx",
    "ss-n": "ss", "ss-i": "ss", "ss-u": "ss", "ss-s": "ss",
}
CLASSES = sorted(BST.keys())
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
TOP_LEVELS = sorted(set(BST.values()))
TOP2IDX = {t: i for i, t in enumerate(TOP_LEVELS)}
TOP_TO_CHILDREN_IDX = {
    top: [CLS2IDX[c] for c in CLASSES if BST[c] == top]
    for top in TOP_LEVELS
}

# CLAP expects 48 kHz mono. BSD10k wavs are 44.1 kHz -> resample.
CLAP_SR = 48000
NATIVE_SR = 44100
# Cap clip length so a batch fits in memory. CLAP was trained on ~10 s windows.
MAX_SAMPLES = CLAP_SR * 10  # 10 seconds at 48 kHz


# ──────────────────────────────────────────────────────────────────────
# Metric + hierarchy-aware predict (identical to v6)
# ──────────────────────────────────────────────────────────────────────
def hierarchical_f(true_labels, pred_labels, lam=0.75):
    total = 1.0 + lam
    hp_per_class = defaultdict(list)
    hr_per_class = defaultdict(list)
    for true, pred in zip(true_labels, pred_labels):
        true_set = {true: 1.0, BST[true]: lam}
        pred_set = {pred: 1.0, BST[pred]: lam}
        overlap = sum(min(true_set.get(k, 0), pred_set.get(k, 0))
                      for k in set(true_set) | set(pred_set))
        hp_per_class[pred].append(overlap / total)
        hr_per_class[true].append(overlap / total)
    all_hp, all_hr = [], []
    for cls in CLASSES:
        hp = np.mean(hp_per_class[cls]) if hp_per_class[cls] else 0.0
        hr = np.mean(hr_per_class[cls]) if hr_per_class[cls] else 0.0
        all_hp.append(hp)
        all_hr.append(hr)
    hp = float(np.mean(all_hp))
    hr = float(np.mean(all_hr))
    hf = 2 * hp * hr / (hp + hr) if (hp + hr) > 0 else 0.0
    return hp, hr, hf


def hierarchy_aware_predict(logits):
    probs = F.softmax(logits, dim=1)
    top_probs = torch.zeros(logits.size(0), len(TOP_LEVELS), device=logits.device)
    for top in TOP_LEVELS:
        children_idx = TOP_TO_CHILDREN_IDX[top]
        top_probs[:, TOP2IDX[top]] = probs[:, children_idx].sum(dim=1)
    adjusted = probs.clone()
    for i, cls in enumerate(CLASSES):
        parent_idx = TOP2IDX[BST[cls]]
        adjusted[:, i] = probs[:, i] * top_probs[:, parent_idx]
    return adjusted.argmax(dim=1)


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.ls)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ──────────────────────────────────────────────────────────────────────
# Dataset: returns RAW WAVEFORM (resampled) + frozen precomputed TEXT emb
# ──────────────────────────────────────────────────────────────────────
class SoundDataset(Dataset):
    def __init__(self, df, dataset_path):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(dataset_path) / "audio"
        self.text_dir  = Path(dataset_path) / "features" / "clap_text_embeddings"
        self.resampler = torchaudio.transforms.Resample(NATIVE_SR, CLAP_SR)

    def __len__(self):
        return len(self.df)

    def _load_wav(self, sound_id):
        path = self.audio_dir / f"{sound_id}.wav"
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:                       # stereo -> mono
            wav = wav.mean(axis=1)
        wav = torch.from_numpy(wav)
        if sr != CLAP_SR:
            if sr == NATIVE_SR:
                wav = self.resampler(wav)
            else:                              # rare odd rate -> per-file resample
                wav = torchaudio.transforms.Resample(sr, CLAP_SR)(wav)
        # pad / truncate to fixed length so we can batch
        if wav.numel() >= MAX_SAMPLES:
            wav = wav[:MAX_SAMPLES]
        else:
            wav = F.pad(wav, (0, MAX_SAMPLES - wav.numel()))
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav  = self._load_wav(row.sound_id)
        text = np.load(self.text_dir / f"{row.sound_id}.npy").astype(np.float32)
        label = CLS2IDX[row["class"]]
        return wav, torch.tensor(text), torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────
# Model: live CLAP audio encoder (partially unfrozen) + frozen text + MLP
# ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512,  256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256,  128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128,   23),
        )

    def forward(self, x):
        return self.layers(x)


class Classifier(pl.LightningModule):
    def __init__(self, lr=1e-3, encoder_lr=3e-6, gamma=2.0,
                 label_smoothing=0.1, warmup_epochs=2, verbose=False):
        super().__init__()
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.warmup_epochs = warmup_epochs
        self.loss_fn = FocalLossWithSmoothing(gamma=gamma, label_smoothing=label_smoothing)
        self.verbose = verbose
        self.val_outputs = []

        # --- load CLAP, freeze everything ---
        self.clap = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap.load_ckpt()  # downloads default HTSAT-tiny checkpoint
        for p in self.clap.parameters():
            p.requires_grad = False

        # --- find the audio branch; we fine-tune ALL of it (31M params,
        #     small enough). No fragile block-index matching. ---
        self.audio_branch = self.clap.model.audio_branch
        self.head = MLP()

        # Start in WARMUP: encoder stays frozen for the first `warmup_epochs`
        # so the randomly-initialized head can settle before its large
        # gradients reach (and potentially wreck) the pretrained encoder.
        self._encoder_unfrozen = False
        n_audio = sum(p.numel() for p in self.audio_branch.parameters())
        print(f"  audio branch = {n_audio/1e6:.1f}M params "
              f"(full fine-tune, frozen for first {warmup_epochs} warmup epochs)")

    def _unfreeze_encoder(self):
        for p in self.audio_branch.parameters():
            p.requires_grad = True
        # Keep ALL normalization layers frozen + in eval mode. With a tiny
        # real batch (8), live BatchNorm statistics are too noisy and drift,
        # which destabilizes fine-tuning. Freeze BN/LN params and stop them
        # updating running stats.
        n_bn = 0
        for m in self.audio_branch.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                n_bn += 1
        self._encoder_unfrozen = True
        n = sum(p.numel() for p in self.audio_branch.parameters() if p.requires_grad)
        print(f"  >> warmup done: unfroze audio branch ({n/1e6:.1f}M trainable, "
              f"{n_bn} norm layers kept frozen)")

    def encode_audio(self, wav):
        # laion_clap exposes get_audio_embedding_from_data; it expects a
        # float tensor [B, samples] at 48 kHz. use_tensor keeps grad flowing.
        emb = self.clap.get_audio_embedding_from_data(x=wav, use_tensor=True)
        return emb  # [B, 512]

    def forward(self, wav, text):
        audio = self.encode_audio(wav)          # live; encoder trains after warmup
        x = torch.cat([audio, text], dim=1)     # 512 + 512
        return self.head(x)

    def configure_optimizers(self):
        # Reference the FULL audio branch in the optimizer from the start.
        # During warmup these params have requires_grad=False, so they get
        # zero gradient and don't move; once warmup ends we flip the flag
        # and they begin training without needing to rebuild the optimizer.
        enc_params  = list(self.audio_branch.parameters())
        head_params = list(self.head.parameters())
        optimizer = torch.optim.Adam(
            [
                {"params": head_params, "lr": self.lr},
                {"params": enc_params,  "lr": self.encoder_lr},
            ],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs,
        )
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        # End of warmup: unfreeze the encoder so it starts adapting.
        if (not self._encoder_unfrozen) and self.current_epoch >= self.warmup_epochs:
            self._unfreeze_encoder()
        # Each epoch Lightning calls model.train(), which re-enables BN/LN
        # train mode. Re-assert eval on the frozen norm layers right away.
        if self._encoder_unfrozen:
            for m in self.audio_branch.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                    m.eval()

    def training_step(self, batch, _):
        wav, text, y = batch
        loss = self.loss_fn(self(wav, text), y)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        wav, text, y = batch
        logits = self(wav, text)
        loss = self.loss_fn(logits, y)
        self.val_outputs.append({
            "loss":   loss.detach().cpu(),
            "logits": logits.detach().cpu(),
            "labels": y.cpu(),
        })

    def on_validation_epoch_end(self):
        all_loss   = torch.stack([o["loss"]   for o in self.val_outputs])
        all_logits = torch.cat ([o["logits"] for o in self.val_outputs])
        all_labels = torch.cat ([o["labels"] for o in self.val_outputs])
        true_str = [IDX2CLS[l.item()] for l in all_labels]

        preds_orig = all_logits.argmax(dim=1)
        pred_orig_str = [IDX2CLS[p.item()] for p in preds_orig]
        hp_o, hr_o, hf_o = hierarchical_f(true_str, pred_orig_str)

        preds_hier = hierarchy_aware_predict(all_logits)
        pred_hier_str = [IDX2CLS[p.item()] for p in preds_hier]
        hp_h, hr_h, hf_h = hierarchical_f(true_str, pred_hier_str)

        self.log_dict({
            "val/loss":    all_loss.mean().item(),
            "val/hF":      hf_o,
            "val/hF_hier": hf_h,
        })

        if self.verbose or (self.current_epoch + 1) % 5 == 0:
            print(f"    epoch {self.current_epoch+1:3d}  "
                  f"loss={all_loss.mean():.4f}  "
                  f"hF={hf_o:.4f}  hF_hier={hf_h:.4f}")

        self.val_outputs.clear()


def run(dataset_path, n_epochs=15, batch_size=8, seed=42,
        early_stop_patience=5, gamma=2.0, label_smoothing=0.1,
        lr=1e-3, encoder_lr=3e-6, warmup_epochs=2, accum_steps=4,
        only_fold=None, run_tag="v8", verbose=False):
    pl.seed_everything(seed, workers=True)

    df = pd.read_csv(Path(dataset_path) / "metadata" / "BSD10k_metadata.csv")
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    print(f"loaded {len(df)} sounds, {df['class'].nunique()} classes")
    print(f"config: FINE-TUNE full CLAP audio branch + frozen text")
    print(f"        head_lr={lr}, encoder_lr={encoder_lr}, warmup={warmup_epochs}ep, "
          f"batch={batch_size}x{accum_steps}accum (eff={batch_size*accum_steps}), fp32, "
          f"max_epochs={n_epochs}, patience={early_stop_patience}")
    if only_fold is not None:
        print(f"        >> RUNNING FOLD {only_fold} ONLY (decision-gate mode)")
    print()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["class"]), 1):
        if only_fold is not None and fold != only_fold:
            continue
        print(f"\n── fold {fold}/5 ──────────────────────────────")
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        train_dl = DataLoader(
            SoundDataset(train_df, dataset_path),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
        )
        val_dl = DataLoader(
            SoundDataset(val_df, dataset_path),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        )

        logger = WandbLogger(project="dcase2026-task1", name=f"{run_tag}_fold{fold}")
        early_stop = EarlyStopping(monitor="val/hF_hier", patience=early_stop_patience,
                                   mode="max", verbose=False)

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            logger=logger,
            accelerator="auto",
            devices=1,
            precision="32-true",                # fp32: TITAN X (Maxwell) has no
                                                # bf16, and fp16 caused NaN loss
            gradient_clip_val=1.0,              # extra NaN insurance for HTSAT
            accumulate_grad_batches=accum_steps, # small real batch, larger effective
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[early_stop],
            log_every_n_steps=10,
        )

        model = Classifier(lr=lr, encoder_lr=encoder_lr, gamma=gamma,
                           label_smoothing=label_smoothing,
                           warmup_epochs=warmup_epochs, verbose=verbose)
        trainer.fit(model, train_dl, val_dl)

        hf_orig = trainer.callback_metrics.get("val/hF",      torch.tensor(0)).item()
        hf_hier = trainer.callback_metrics.get("val/hF_hier", torch.tensor(0)).item()
        stop_epoch = trainer.current_epoch + 1

        results.append({"fold": fold, "hF": hf_orig, "hF_hier": hf_hier, "stop_epoch": stop_epoch})
        print(f"  fold {fold} done after {stop_epoch} epochs   hF={hf_orig:.4f}   hF_hier={hf_hier:.4f}")
        logger.experiment.finish()

        del model, trainer
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"SUMMARY ({run_tag})")
    print(f"{'='*60}")
    print(f"{'Fold':<6}{'Stop@':<8}{'hF (argmax)':<15}{'hF (hier)':<15}")
    print(f"{'-'*44}")
    for r in results:
        print(f"{r['fold']:<6}{r['stop_epoch']:<8}{r['hF']:<15.4f}{r['hF_hier']:<15.4f}")
    print(f"{'-'*44}")
    mean_hf      = np.mean([r['hF']      for r in results])
    mean_hf_hier = np.mean([r['hF_hier'] for r in results])
    std_hf       = np.std ([r['hF']      for r in results])
    std_hf_hier  = np.std ([r['hF_hier'] for r in results])
    print(f"{'MEAN':<6}{'':<8}{mean_hf:<15.4f}{mean_hf_hier:<15.4f}")
    print(f"{'STD':<6}{'':<8}{std_hf:<15.4f}{std_hf_hier:<15.4f}")
    print(f"{'='*60}")
    print(f"compare against:")
    print(f"  v6 best (frozen):  0.8048")
    print(f"  v1 baseline:       0.8013")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=3e-6)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--fold", type=int, default=None,
                        help="run only this fold (1-5) for decision-gate mode")
    parser.add_argument("--tag", type=str, default="v10")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run(args.dataset_path, n_epochs=args.epochs, batch_size=args.batch_size,
        early_stop_patience=args.patience, gamma=args.gamma,
        label_smoothing=args.label_smoothing, lr=args.lr,
        encoder_lr=args.encoder_lr, warmup_epochs=args.warmup_epochs,
        accum_steps=args.accum_steps, only_fold=args.fold,
        run_tag=args.tag, verbose=args.verbose)
