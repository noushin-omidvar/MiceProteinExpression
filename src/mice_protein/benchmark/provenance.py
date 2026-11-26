"""Record local checkpoint identities without triggering a network download."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

TABPFN_V2_SHA256 = "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
TABPFN_V2_FILENAME = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
KUMO_SMALL_SHA256 = "1ff91f484e19021aaf7d9eff6cc07e6d95a60b4aaad473e83e2ab2c3bfd2b617"
KUMO_REVISION = "3c3e10bbdb590ace29e7026847f92db3c603096d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_inventory() -> dict:
    home = Path.home()
    found = {}
    try:
        tabpfn = pinned_tabpfn_v2_path()
        found["tabpfn_v2"] = {"filename": tabpfn.name,
                              "sha256": _sha256(tabpfn)}
    except FileNotFoundError:
        pass
    hub_base = Path(os.environ.get("HF_HUB_CACHE", str(home / ".cache/huggingface/hub")))
    hub = hub_base / "models--nvidia--Kumo-Tabular/snapshots"
    kumo = sorted(hub.glob(f"{KUMO_REVISION}/small/classifier.pt"))
    if kumo:
        found["kumo_tabular_small"] = {"huggingface_revision": kumo[0].parents[1].name,
                                       "filename": "small/classifier.pt",
                                       "sha256": _sha256(kumo[0])}
    return found


def pinned_tabpfn_v2_path() -> Path:
    """Resolve and verify the exact v2 file used in the reported experiments."""
    override = os.environ.get("MICE_TABPFN_V2_CHECKPOINT")
    candidates = ([Path(override)] if override else [
        Path.home() / "Library/Caches/tabpfn" / TABPFN_V2_FILENAME,
        Path.home() / ".cache/tabpfn" / TABPFN_V2_FILENAME,
    ])
    for path in candidates:
        if path.is_file():
            if _sha256(path) != TABPFN_V2_SHA256:
                raise RuntimeError(f"TabPFN v2 checkpoint hash mismatch: {path}")
            return path
    raise FileNotFoundError(
        "Pinned TabPFN v2 checkpoint not found; set MICE_TABPFN_V2_CHECKPOINT "
        f"to a copy of {TABPFN_V2_FILENAME} with SHA-256 {TABPFN_V2_SHA256}"
    )


def verify_kumo_small_cache() -> None:
    """Verify the file loaded by NVIDIA's v1.0.1 cache-backed constructor."""
    hub = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    path = (hub / "models--nvidia--Kumo-Tabular/snapshots" /
            KUMO_REVISION / "small/classifier.pt")
    if not path.is_file() or _sha256(path) != KUMO_SMALL_SHA256:
        raise RuntimeError("Kumo Small v1.0.1 cached checkpoint is missing or has changed")
