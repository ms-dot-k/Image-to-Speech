import glob
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import torch
from tqdm import tqdm

from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import (
    HubertFeatureReader,
)


wavs_dir = "?"
out_dir = "?"

# Global variables for worker initialization
_kmeans_model = None
_fr = None


def _init_worker(kmeans_path: str, hubert_path: str, hubert_layer: int):
    """Initialize worker process with models."""
    global _kmeans_model, _fr
    _kmeans_model = joblib.load(kmeans_path)
    _fr = HubertFeatureReader(hubert_path, hubert_layer)


def process_file(file):
    """Process a single WAV file and save Hubert units."""
    if not file.endswith(".wav"):
        return None

    rel_path = os.path.relpath(file, wavs_dir)
    out_path = os.path.join(out_dir, rel_path)
    out = Path(out_path)
    out_path = str(out.with_suffix(".unit"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Read features and predict clusters using global worker models
    feat = _fr.get_feats(file)
    feat = feat.cpu()
    _kmeans_model.cluster_centers_ = _kmeans_model.cluster_centers_.astype(np.float64)
    pred = _kmeans_model.predict(feat)
    pred = np.asarray(pred, dtype=np.int64)
    torch.save(pred, out_path)
    # print(f"Processed {file} --> {out_path}")

    return file


def list_files_glob(pattern=None, recursive=True, num_workers=4):
    """Process WAV files using multiple processes with shared models."""
    if pattern is None:
        pattern = os.path.join(wavs_dir, "**/*")
    files = glob.glob(pattern, recursive=recursive)
    wav_files = [f for f in files if f.endswith(".wav")]

    print(f"Found {len(wav_files)} WAV files. Processing with {num_workers} workers...")

    # Use initializer to load models once per worker process
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=("km.bin", "hubert_base_ls960.pt", 6),
    ) as executor:
        # Submit all tasks and track them
        future_to_file = {executor.submit(process_file, f): f for f in wav_files}

        # Process results with progress bar
        for future in tqdm(future_to_file, total=len(wav_files), desc="Processing"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {future_to_file[future]}: {e}")


if __name__ == "__main__":
    list_files_glob()
