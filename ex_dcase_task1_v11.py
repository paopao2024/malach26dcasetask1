"""
DCASE 2026 Task 1 - v11 (proper fine-tune: waveform+spec aug, layer-wise LR, freeze BN only, longer patience)
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
    def __init__(self, df, dataset_path, augment=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(dataset_path) / "audio"
        self.text_dir  = Path(dataset_path) / "features" / "clap_text_embeddings"
        self.resampler = torchaudio.transforms.Resample(NATIVE_SR, CLAP_SR)
        self.augment = augment           # waveform aug only on the training set

    def __len__(self):
        return len(self.df)

    def _augment_wav(self, wav):
        # Light waveform augmentation: random circular time-shift + gaussian noise.
        # (SpecAugment is applied separately inside HTSAT during training.)
        if torch.rand(1).item() < 0.5:
            shift = int(torch.randint(0, wav.numel(), (1,)).item())
            wav = torch.roll(wav, shift)
        if torch.rand(1).item() < 0.5:
            snr_noise = 0.005 * torch.randn_like(wav)
            wav = wav + snr_noise
        return wav

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
        if self.augment:
            wav = self._augment_wav(wav)
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

        # --- load CLAP, freeze everything ---
        self.clap = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap.load_ckpt()  # downloads default HTSAT-tiny checkpoint
        for p in self.clap.parameters():
            p.requires_grad = False

        self.audio_branch = self.clap.model.audio_branch
        # HTSAT-tiny has 4 Swin stages in .layers; we apply layer-wise LR
        # across them (top stage trains fastest, bottom slowest).
        self.swin_stages = list(self.audio_branch.layers)
        self.head = MLP()

        # Start in WARMUP: encoder frozen so the random head settles first.
        self._encoder_unfrozen = False
        n_audio = sum(p.numel() for p in self.audio_branch.parameters())
        print(f"  audio branch = {n_audio/1e6:.1f}M params, "
              f"{len(self.swin_stages)} Swin stages "
              f"(frozen for first {warmup_epochs} warmup epochs)")

    def _freeze_batchnorm_only(self):
        """Freeze ONLY BatchNorm (1 layer: bn0) — its running stats are
        unreliable with a small batch. LayerNorms (29) are batch-independent
        and SHOULD train, so we leave them on. This corrects the v9/v10
        mistake of freezing all norms."""
        n_bn = 0
        for m in self.audio_branch.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                n_bn += 1
        return n_bn

    def _unfreeze_encoder(self):
        for p in self.audio_branch.parameters():
            p.requires_grad = True
        n_bn = self._freeze_batchnorm_only()
        self._encoder_unfrozen = True
        n = sum(p.numel() for p in self.audio_branch.parameters() if p.requires_grad)
        print(f"  >> warmup done: unfroze audio branch "
              f"({n/1e6:.1f}M trainable, {n_bn} BatchNorm frozen, LayerNorms train)")

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
        # Layer-wise discriminative LRs across the 4 Swin stages.
        # Top stage (index -1) gets encoder_lr_top; each lower stage is
        # multiplied by lr_decay (so bottom stages barely move — they hold
        # general audio features that shouldn't be disturbed).
        param_groups = [{"params": list(self.head.parameters()), "lr": self.lr}]
        n_stages = len(self.swin_stages)
        stage_param_ids = set()
        for depth, stage in enumerate(self.swin_stages):
            # depth 0 = bottom (slowest), depth n-1 = top (fastest)
            stage_lr = self.encoder_lr_top * (self.lr_decay ** (n_stages - 1 - depth))
            sp = list(stage.parameters())
            param_groups.append({"params": sp, "lr": stage_lr})
            stage_param_ids.update(id(p) for p in sp)

        # remaining audio-branch params not inside a Swin stage
        # (patch_embed, bn0, norm, tscam_conv, head of HTSAT, etc.) get the
        # top LR as well.
        rest = [p for p in self.audio_branch.parameters()
                if id(p) not in stage_param_ids]
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
            for m in self.audio_branch.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
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


def run(dataset_path, n_epochs=30, batch_size=8, seed=42,
        early_stop_patience=10, gamma=2.0, label_smoothing=0.1,
        lr=1e-3, encoder_lr_top=3e-5, lr_decay=0.5, warmup_epochs=2,
        accum_steps=4, only_fold=None, run_tag="v11", verbose=False):
    pl.seed_everything(seed, workers=True)

    df = pd.read_csv(Path(dataset_path) / "metadata" / "BSD10k_metadata.csv")
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    print(f"loaded {len(df)} sounds, {df['class'].nunique()} classes")
    print(f"config: PROPER FINE-TUNE full CLAP audio + frozen text")
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
    print(f"compare against:")
    print(f"  v6 best (frozen):  0.8048")
    print(f"  v1 baseline:       0.8013")
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
                        help="LR for the TOP Swin stage; lower stages decay")
    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="per-stage LR multiplier going downward")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--fold", type=int, default=None,
                        help="run only this fold (1-5) for decision-gate mode")
    parser.add_argument("--tag", type=str, default="v11")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run(args.dataset_path, n_epochs=args.epochs, batch_size=args.batch_size,
        early_stop_patience=args.patience, gamma=args.gamma,
        label_smoothing=args.label_smoothing, lr=args.lr,
        encoder_lr_top=args.encoder_lr_top, lr_decay=args.lr_decay,
        warmup_epochs=args.warmup_epochs,
        accum_steps=args.accum_steps, only_fold=args.fold,
        run_tag=args.tag, verbose=args.verbose)
