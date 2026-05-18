"""
Extract PaSST embeddings for BSD10k audio files.

Loads each audio file, resamples to 32kHz, crops/pads to exactly 10 seconds,
runs through PaSST, and saves a 768-dim embedding as .npy.

Supports resume: if interrupted, just re-run and it'll skip files already done.
"""

import warnings
import os
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from hear21passt.base import get_basic_model


def load_audio_clip(path, target_sr=32000, target_samples=320000):
    """Load audio, mono, resample to 32kHz, crop/pad to exactly 10 sec."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[1] >= target_samples:
        wav = wav[:, :target_samples]
    else:
        pad = target_samples - wav.shape[1]
        wav = torch.nn.functional.pad(wav, (0, pad))
    return wav  # shape [1, 320000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="folder containing .wav files named <sound_id>.wav")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="folder to save <sound_id>.npy embeddings")
    parser.add_argument("--metadata_csv", type=str, required=True,
                        help="BSD10k_metadata.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading PaSST model...")
    model = get_basic_model(mode="embed_only")
    model.eval()
    model.to(device)

    df = pd.read_csv(args.metadata_csv)
    sound_ids = df["sound_id"].astype(str).tolist()
    print(f"total sounds in metadata: {len(sound_ids)}")

    # resume: skip files already extracted
    todo = [sid for sid in sound_ids
            if not (output_dir / f"{sid}.npy").exists()]
    already = len(sound_ids) - len(todo)
    print(f"already extracted: {already}")
    print(f"to extract: {len(todo)}")

    if not todo:
        print("nothing to do, all extracted.")
        return

    extracted = 0
    failed = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(todo), args.batch_size), desc="batches"):
            batch_ids = todo[i:i + args.batch_size]
            wavs = []
            valid_ids = []

            for sid in batch_ids:
                audio_file = audio_dir / f"{sid}.wav"
                if not audio_file.exists():
                    failed += 1
                    continue
                try:
                    wavs.append(load_audio_clip(audio_file))
                    valid_ids.append(sid)
                except Exception as e:
                    print(f"\n  load failed {sid}: {e}")
                    failed += 1

            if not wavs:
                continue

            try:
                batch_tensor = torch.cat(wavs, dim=0).to(device)
                embeddings = model(batch_tensor)
                for j, sid in enumerate(valid_ids):
                    emb_np = embeddings[j].cpu().numpy().astype(np.float32)
                    np.save(output_dir / f"{sid}.npy", emb_np)
                    extracted += 1
            except Exception as e:
                print(f"\n  batch inference failed: {e}")
                failed += len(valid_ids)

    print(f"\n{'='*50}")
    print(f"extraction done")
    print(f"  extracted this run: {extracted}")
    print(f"  failed:             {failed}")
    print(f"  total now on disk:  {already + extracted}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
