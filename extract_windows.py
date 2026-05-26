"""
Multi-crop windowed CLAP embedding extraction.

The standard pipeline pools each (up to 30s) clip into ONE 512-d CLAP audio
embedding -> for long clips this averages away most temporal information.
Nearly half of BSD10k clips are >10s (mean 14.8s, max 30s), so a lot of signal
is being thrown away.

Here we slice each clip into overlapping windows, embed each window through
FROZEN CLAP, and save one .npy per window. Downstream training then treats
each window as a training example (with the clip's label) and aggregates
windows at inference -> more data + finer temporal resolution.

Output layout (mirrors the existing features dir so downstream is easy):
  <out_dir>/clap_audio_win/<sound_id>__w{k}.npy     # one per window
  <out_dir>/window_index.csv                        # sound_id, n_windows

Text embeddings are unchanged (one per clip) -- we reuse the existing
clap_text_embeddings and just broadcast the clip's text to each window.

This is a FROZEN forward pass (no training). It is the slow one-time step;
everything after is fast frozen-feature work.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import soundfile as sf
import torchaudio
import laion_clap

CLAP_SR = 48000
NATIVE_SR = 44100


def slice_windows(wav, sr, win_sec=7.0, hop_sec=3.5):
    """Return a list of fixed-length windows (samples). Overlapping.
    Short clips (< win) produce a single padded window."""
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    n = wav.numel()
    if n <= win:
        return [F.pad(wav, (0, win - n))]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:                 # tail window to cover the end
        starts.append(n - win)
    return [wav[s:s + win] for s in starts]


def load_wav(path, resampler):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav)
    if sr != CLAP_SR:
        if sr == NATIVE_SR:
            wav = resampler(wav)
        else:
            wav = torchaudio.transforms.Resample(sr, CLAP_SR)(wav)
    return wav


def main(dataset_path, out_dir, win_sec, hop_sec, limit=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_dir = Path(dataset_path) / "audio"
    out_dir = Path(out_dir)
    win_dir = out_dir / "clap_audio_win"
    win_dir.mkdir(parents=True, exist_ok=True)

    print("loading CLAP...")
    clap = laion_clap.CLAP_Module(enable_fusion=False)
    clap.load_ckpt()
    clap.eval().to(device)
    for p in clap.parameters():
        p.requires_grad = False

    resampler = torchaudio.transforms.Resample(NATIVE_SR, CLAP_SR)

    wavs = sorted(audio_dir.glob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    print(f"extracting windowed embeddings for {len(wavs)} clips "
          f"(win={win_sec}s hop={hop_sec}s)...")

    index_rows = []
    done = 0
    with torch.no_grad():
        for wav_path in wavs:
            sid = wav_path.stem
            wav = load_wav(str(wav_path), resampler)
            windows = slice_windows(wav, CLAP_SR, win_sec, hop_sec)
            # batch all windows of this clip through CLAP at once
            batch = torch.stack(windows).to(device)            # [W, win_samples]
            emb = clap.get_audio_embedding_from_data(x=batch, use_tensor=True)  # [W, 512]
            emb = emb.cpu().numpy().astype(np.float32)
            for k in range(emb.shape[0]):
                np.save(win_dir / f"{sid}__w{k}.npy", emb[k])
            index_rows.append({"sound_id": sid, "n_windows": emb.shape[0]})
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(wavs)} clips done")

    idx_df = pd.DataFrame(index_rows)
    idx_df.to_csv(out_dir / "window_index.csv", index=False)
    print(f"\ndone. {done} clips -> {idx_df['n_windows'].sum()} windows")
    print(f"avg windows/clip: {idx_df['n_windows'].mean():.2f}")
    print(f"index -> {out_dir/'window_index.csv'}")
    print(f"windows -> {win_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="default: <dataset_path>/features")
    parser.add_argument("--win_sec", type=float, default=7.0)
    parser.add_argument("--hop_sec", type=float, default=3.5)
    parser.add_argument("--limit", type=int, default=None,
                        help="only first N clips (for a quick test run)")
    args = parser.parse_args()
    out = args.out_dir or str(Path(args.dataset_path) / "features")
    main(args.dataset_path, out, args.win_sec, args.hop_sec, args.limit)
