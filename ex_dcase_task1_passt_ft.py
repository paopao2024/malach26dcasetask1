"""
DCASE 2026 Task 1 - PaSST audio-only fine-tune
Tests professors' suggestion #2: fine-tune PaSST on the data, ignoring text,
to see if PaSST alone is strong enough.

Design:
- Load raw .wav -> resample to 32 kHz (PaSST's rate) -> live PaSST encoder
  (hear21passt, mode='embed_only') -> 768-d audio embedding -> MLP -> 23 classes.
- NO TEXT. We deliberately do not fuse CLAP text: PaSST and CLAP-text live in
  unrelated embedding spaces, so gluing them is not meaningful. This is a clean
  audio-only test.
- Comparison baseline is FROZEN PaSST-only (~0.74), NOT the 0.8048 multimodal CLAP.
  The question: does fine-tuning lift PaSST above its own frozen 0.74?

Carries over all the lessons from the CLAP attempts (v8-v11):
- 2-epoch head warmup, then unfreeze the encoder.
- Layer-wise LR across PaSST's 12 transformer blocks.
- Freeze BatchNorm only (small-batch drift); train LayerNorms.
- Waveform augmentation + PaSST's built-in SpecAugment (train mode).
- fp32, grad-clip, batch 8 x4 accum, longer patience, fold-1 decision gate.
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
from hear21passt.base import get_basic_model


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

# PaSST expects 32 kHz mono. BSD10k wavs are 44.1 kHz -> resample.
PASST_SR = 32000
NATIVE_SR = 44100
# Cap clip length so a batch fits in memory. 10 s at 32 kHz.
MAX_SAMPLES = PASST_SR * 10  # 320000 samples


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
# Dataset: returns RAW WAVEFORM (resampled to 32 kHz). Audio-only, no text.
# ──────────────────────────────────────────────────────────────────────
class SoundDataset(Dataset):
    def __init__(self, df, dataset_path, augment=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(dataset_path) / "audio"
        self.resampler = torchaudio.transforms.Resample(NATIVE_SR, PASST_SR)
        self.augment = augment           # waveform aug only on the training set

    def __len__(self):
        return len(self.df)

    def _augment_wav(self, wav):
        # Light waveform augmentation: random circular time-shift + gaussian noise.
        # (SpecAugment is applied separately inside PaSST during training.)
        if torch.rand(1).item() < 0.5:
            shift = int(torch.randint(0, wav.numel(), (1,)).item())
            wav = torch.roll(wav, shift)
        if torch.rand(1).item() < 0.5:
            wav = wav + 0.005 * torch.randn_like(wav)
        return wav

    def _load_wav(self, sound_id):
        path = self.audio_dir / f"{sound_id}.wav"
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:                       # stereo -> mono
            wav = wav.mean(axis=1)
        wav = torch.from_numpy(wav)
        if sr != PASST_SR:
            if sr == NATIVE_SR:
                wav = self.resampler(wav)
            else:                              # rare odd rate -> per-file resample
                wav = torchaudio.transforms.Resample(sr, PASST_SR)(wav)
        # pad / truncate to fixed length so we can batch
        if wav.numel() >= MAX_SAMPLES:
            wav = wav[:MAX_SAMPLES]
        else:
            wav = F.pad(wav, (0, MAX_SAMPLES - wav.numel()))
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav  = self._load_wav(row.sound_id)
        if self.augment:
            wav = self._augment_wav(wav)
        label = CLS2IDX[row["class"]]
        return wav, torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────
# Model: live PaSST encoder (fine-tuned) -> MLP. Audio-only.
# ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512,    256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256,    128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128,     23),
        )

    def forward(self, x):
        return self.layers(x)


class Classifier(pl.LightningModule):
    def __init__(self, lr=1e-3, encoder_lr_top=3e-5, lr_decay=0.5, gamma=2.0,
                 label_smoothing=0.1, warmup_epochs=2, verbose=False):
        super().__init__()
        self.lr = lr                      # MLP head LR
        self.encoder_lr_top = encoder_lr_top   # LR for the TOP Swin stage
        self.lr_decay = lr_decay          # each lower stage gets lr * decay^depth
        self.warmup_epochs = warmup_epochs
        self.loss_fn = FocalLossWithSmoothing(gamma=gamma, label_smoothing=label_smoothing)
        self.verbose = verbose
        self.val_outputs = []

        # --- load PaSST, freeze everything ---
        self.passt = get_basic_model(mode="embed_only")
        for p in self.passt.parameters():
            p.requires_grad = False

        # The actual transformer lives at self.passt.net. Its blocks are in
        # .net.blocks (12 of them). We apply layer-wise LR across those blocks.
        self.passt_net = self.passt.net
        self.blocks = self._find_blocks()
        self.head = MLP(in_dim=768)

        # Start in WARMUP: encoder frozen so the random head settles first.
        self._encoder_unfrozen = False
        n_audio = sum(p.numel() for p in self.passt.parameters())
        print(f"  PaSST = {n_audio/1e6:.1f}M params, "
              f"{len(self.blocks)} transformer blocks "
              f"(frozen for first {warmup_epochs} warmup epochs)")

    def _find_blocks(self):
        """Locate PaSST's transformer block list for layer-wise LR. The
        hear21passt model nests the ViT-style encoder at .net; blocks are at
        .net.blocks. Fall back gracefully if the attribute path differs."""
        net = self.passt_net
        if hasattr(net, "blocks"):
            return list(net.blocks)
        # defensive fallback: search for a ModuleList of repeated blocks
        for m in net.modules():
            if isinstance(m, nn.ModuleList) and len(m) >= 6:
                return list(m)
        return []

    def _freeze_batchnorm_only(self):
        """Freeze ONLY BatchNorm — running stats are unreliable with a small
        batch. LayerNorms are batch-independent and SHOULD train."""
        n_bn = 0
        for m in self.passt.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                n_bn += 1
        return n_bn

    def _unfreeze_encoder(self):
        for p in self.passt.parameters():
            p.requires_grad = True
        n_bn = self._freeze_batchnorm_only()
        self._encoder_unfrozen = True
        n = sum(p.numel() for p in self.passt.parameters() if p.requires_grad)
        print(f"  >> warmup done: unfroze PaSST "
              f"({n/1e6:.1f}M trainable, {n_bn} BatchNorm frozen, LayerNorms train)")

    def encode_audio(self, wav):
        # hear21passt embed_only model takes a waveform tensor [B, samples]
        # at 32 kHz and returns a [B, 768] embedding. Grad flows through it.
        return self.passt(wav)  # [B, 768]

    def forward(self, wav):
        audio = self.encode_audio(wav)          # live; encoder trains after warmup
        return self.head(audio)                 # audio-only, no text

    def configure_optimizers(self):
        # Layer-wise discriminative LRs across PaSST's transformer blocks.
        # Top block gets encoder_lr_top; each lower block is multiplied by
        # lr_decay (bottom blocks barely move — they hold general features).
        param_groups = [{"params": list(self.head.parameters()), "lr": self.lr}]
        n_blocks = len(self.blocks)
        block_param_ids = set()
        for depth, block in enumerate(self.blocks):
            stage_lr = self.encoder_lr_top * (self.lr_decay ** (n_blocks - 1 - depth))
            sp = list(block.parameters())
            param_groups.append({"params": sp, "lr": stage_lr})
            block_param_ids.update(id(p) for p in sp)

        # remaining PaSST params not inside a block (patch embed, pos embed,
        # cls/dist tokens, final norm, etc.) get the top LR.
        rest = [p for p in self.passt.parameters()
                if id(p) not in block_param_ids]
        if rest:
            param_groups.append({"params": rest, "lr": self.encoder_lr_top})

        optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs,
        )
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        # End of warmup: unfreeze encoder so it starts adapting.
        if (not self._encoder_unfrozen) and self.current_epoch >= self.warmup_epochs:
            self._unfreeze_encoder()
        # Re-assert BN eval each epoch (Lightning's model.train() re-enables it).
        # LayerNorms are intentionally left training.
        if self._encoder_unfrozen:
            for m in self.passt.modules():
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


def run(dataset_path, n_epochs=30, batch_size=8, seed=42,
        early_stop_patience=10, gamma=2.0, label_smoothing=0.1,
        lr=1e-3, encoder_lr_top=3e-5, lr_decay=0.5, warmup_epochs=2,
        accum_steps=4, only_fold=None, run_tag="passt_ft", verbose=False):
    pl.seed_everything(seed, workers=True)

    df = pd.read_csv(Path(dataset_path) / "metadata" / "BSD10k_metadata.csv")
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    print(f"loaded {len(df)} sounds, {df['class'].nunique()} classes")
    print(f"config: FINE-TUNE PaSST audio-only (NO text)")
    print(f"        head_lr={lr}, encoder_lr_top={encoder_lr_top}, "
          f"lr_decay={lr_decay} (layer-wise), warmup={warmup_epochs}ep, "
          f"waveform-aug=ON, batch={batch_size}x{accum_steps}accum "
          f"(eff={batch_size*accum_steps}), fp32, max_epochs={n_epochs}, "
          f"patience={early_stop_patience}")
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
            SoundDataset(train_df, dataset_path, augment=True),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
        )
        val_dl = DataLoader(
            SoundDataset(val_df, dataset_path, augment=False),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        )

        logger = WandbLogger(project="dcase2026-task1", name=f"{run_tag}_fold{fold}")
        early_stop = EarlyStopping(monitor="val/hF_hier", patience=early_stop_patience,
                                   mode="max", verbose=False)

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            min_epochs=warmup_epochs + 8,       # don't let early-stop fire during
                                                # the expected fine-tune dip; give
                                                # the encoder room to recover+climb
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

        model = Classifier(lr=lr, encoder_lr_top=encoder_lr_top, lr_decay=lr_decay,
                           gamma=gamma, label_smoothing=label_smoothing,
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
    print(f"compare against (PaSST audio-only is the right baseline here):")
    print(f"  frozen PaSST-only:   ~0.74  <-- does fine-tuning beat THIS?")
    print(f"  (CLAP audio+text:     0.8048, different setup, not comparable)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr_top", type=float, default=3e-5,
                        help="LR for the TOP PaSST block; lower blocks decay")
    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="per-block LR multiplier going downward")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--fold", type=int, default=None,
                        help="run only this fold (1-5) for decision-gate mode")
    parser.add_argument("--tag", type=str, default="passt_ft")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run(args.dataset_path, n_epochs=args.epochs, batch_size=args.batch_size,
        early_stop_patience=args.patience, gamma=args.gamma,
        label_smoothing=args.label_smoothing, lr=args.lr,
        encoder_lr_top=args.encoder_lr_top, lr_decay=args.lr_decay,
        warmup_epochs=args.warmup_epochs,
        accum_steps=args.accum_steps, only_fold=args.fold,
        run_tag=args.tag, verbose=args.verbose)
