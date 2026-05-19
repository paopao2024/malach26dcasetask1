"""
Extract PANNs CNN14 embeddings for BSD10k audio files.

Loads each audio file, resamples to 32kHz mono, crops/pads to 10 seconds,
runs through PANNs CNN14, and saves a 2048-dim embedding as .npy.

Supports resume: re-running skips files already done.
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
from panns_inference import AudioTagging


TARGET_SR = 32000
TARGET_SAMPLES = TARGET_SR * 10  # 10 seconds


def load_audio_clip(path):
    """Load audio, mono, resample to 32kHz, crop/pad to exactly 10 sec."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    if wav.shape[1] >= TARGET_SAMPLES:
        wav = wav[:, :TARGET_SAMPLES]
    else:
        pad = TARGET_SAMPLES - wav.shape[1]
        wav = torch.nn.functional.pad(wav, (0, pad))
    return wav.squeeze(0).numpy().astype(np.float32)  # shape (320000,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("loading PANNs CNN14 model (may download ~300MB checkpoint on first run)...")
    at_model = AudioTagging(checkpoint_path=None, device=device)

    df = pd.read_csv(args.metadata_csv)
    sound_ids = df["sound_id"].astype(str).tolist()
    print(f"total sounds in metadata: {len(sound_ids)}")

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
            # panns_inference expects [batch, samples] numpy
            audio_np = np.stack(wavs)
            # returns (clipwise_output, embedding) tuple
            _, embeddings = at_model.inference(audio_np)
            # embeddings: [batch, 2048]
            for j, sid in enumerate(valid_ids):
                emb = embeddings[j].astype(np.float32)
                np.save(output_dir / f"{sid}.npy", emb)
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
