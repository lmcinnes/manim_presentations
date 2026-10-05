"""
Precompute data for quickdraw_knn_slides.py: Quick, Draw! doodles from a few
visually distinct classes, embedded with DINOv2; nearest neighbours in the
DINOv2 embedding space, 2D UMAP, and ranked showcase edges
(see knn_edge_prep).

    pip install torch transformers umap-learn
    python quickdraw_knn_prep.py
    python quickdraw_knn_prep.py --per-class 3000 --model facebook/dinov2-small

Writes quickdraw_knn.npz and quickdraw_knn.candidates.png (for hand-picking).
The DINOv2 features are cached (quickdraw_cache/), so re-running with
different selection settings (--long-frac, --feature-purity, ...) is quick.
"""

import argparse
import io
import struct
import urllib.request
from pathlib import Path

import numpy as np

from knn_edge_prep import add_analysis_args, analyse_and_save

HERE = Path(__file__).resolve().parent
GCS = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

# Visually distinct doodle classes (different overall shapes and parts), so
# the clusters are clean and any cross-class nearest neighbour is striking.
CATEGORIES = [
    "airplane",
    "bicycle",
    "clock",
    "octopus",
    "violin",
    "mushroom",
    "cloud",
    "ladder",
    "fish",
    "house",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _npy_header(buf):
    """Parse a .npy header -> (header_length, shape)."""
    import ast

    assert buf[:6] == b"\x93NUMPY", "not an .npy file"
    major = buf[6]
    if major == 1:
        hlen = struct.unpack("<H", buf[8:10])[0]
        start = 10
    else:
        hlen = struct.unpack("<I", buf[8:12])[0]
        start = 12
    header = ast.literal_eval(buf[start : start + hlen].decode("latin1"))
    assert header["descr"] in ("|u1", "<u1", "u1"), header
    return start + hlen, header["shape"]


def _range_get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        if r.status != 206:
            raise IOError("server ignored the Range header")
        return r.read()


def fetch_class(cat, n, offset, cache_dir):
    """n bitmaps (uint8, n x 784) from a category, starting at row `offset`.
    Uses HTTP range requests so only the needed bytes are downloaded (the
    full files are ~100-400 MB); falls back to a full download."""
    path = Path(cache_dir) / f"{cat}_{offset}_{n}.npy"
    if path.exists():
        return np.load(path)
    url = GCS.format(cat)
    try:
        head = _range_get(url, 0, 255)
        data_start, shape = _npy_header(head)
        n_total = shape[0]
        n = min(n, n_total - offset)
        a = data_start + offset * 784
        raw = _range_get(url, a, a + n * 784 - 1)
        X = np.frombuffer(raw, np.uint8).reshape(n, 784)
    except Exception as e:
        print(f"  range request failed for {cat} ({e}); full download")
        with urllib.request.urlopen(url) as r:
            X = np.load(io.BytesIO(r.read()))[offset : offset + n]
    np.save(path, X)
    return X


def load_quickdraw(categories, per_class, offset, cache_dir):
    Path(cache_dir).mkdir(exist_ok=True)
    X, y = [], []
    for c, cat in enumerate(categories):
        print("fetching", cat)
        Xc = fetch_class(cat, per_class, offset, cache_dir)
        X.append(Xc)
        y.extend([c] * len(Xc))
    return np.vstack(X), np.array(y)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_dinov2(X, model_id, batch_size, cache_path):
    """L2-normalised DINOv2 [CLS] embeddings of 28x28 doodles (inverted to
    black-on-white RGB, resized to 224)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        z = np.load(cache_path)
        if z["n"] == len(X) and str(z["model"]) == model_id:
            print("using cached features", cache_path)
            return z["features"]

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"embedding {len(X)} images with {model_id} on {device}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    feats = []
    for i in range(0, len(X), batch_size):
        batch = [
            Image.fromarray(255 - img.reshape(28, 28))
            .convert("RGB")
            .resize((224, 224), Image.BILINEAR)
            for img in X[i : i + batch_size]
        ]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            f = (
                out.pooler_output
                if getattr(out, "pooler_output", None) is not None
                else out.last_hidden_state[:, 0, :]
            )
            f = torch.nn.functional.normalize(f, p=2, dim=1)
        feats.append(f.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  {min(i + batch_size, len(X))}/{len(X)}")
    F = np.vstack(feats).astype(np.float32)
    np.savez(cache_path, features=F, n=len(X), model=model_id)
    return F


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES,
        help="at most 10 (one palette colour each)",
    )
    p.add_argument("--per-class", type=int, default=1000)
    p.add_argument(
        "--offset", type=int, default=0, help="first row taken from each class file"
    )
    p.add_argument("--model", default="facebook/dinov2-base")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cache", default=str(HERE / "quickdraw_cache"))
    p.add_argument("--out", default=str(HERE / "quickdraw_knn.npz"))
    add_analysis_args(p)
    a = p.parse_args()
    if len(a.categories) > 10:
        p.error("at most 10 categories (one palette colour each)")

    X, y = load_quickdraw(a.categories, a.per_class, a.offset, a.cache)
    tag = a.model.replace("/", "_")
    F = embed_dinov2(
        X,
        a.model,
        a.batch_size,
        Path(a.cache) / f"features_{tag}_{'-'.join(a.categories)}_"
        f"{a.offset}_{a.per_class}.npz",
    )
    # features are L2-normalised, so Euclidean neighbours = cosine neighbours
    analyse_and_save(
        F, y, a.categories, X, a.out, a, feature_space="DINOv2 embedding space"
    )


if __name__ == "__main__":
    main()
