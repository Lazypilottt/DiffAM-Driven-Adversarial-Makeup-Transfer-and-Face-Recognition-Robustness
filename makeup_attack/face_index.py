"""ArcFace embedding index over a gender-partitioned reference gallery.

Gallery layout::

    reference_pool/
        male/    (or boys/, men/)
        female/  (or girls/, women/)

Embeddings are computed once with InsightFace ``buffalo_l`` and stored in a
single ``.npz`` file. A query is then one matrix-vector product instead of
re-running detection on every gallery image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

GENDER_ALIASES = {
    "male": "M", "males": "M", "man": "M", "men": "M", "boy": "M", "boys": "M", "m": "M",
    "female": "F", "females": "F", "woman": "F", "women": "F", "girl": "F", "girls": "F", "f": "F",
}


@dataclass(frozen=True)
class Match:
    path: str
    similarity: float
    gender: str


def load_analyzer(det_size: int = 640, ctx_id: int = 0):
    """InsightFace detector + ArcFace recogniser + gender/age head."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition", "genderage"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def read_image(path: str | os.PathLike):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img


def largest_face(faces: Sequence):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def embed_query(app, image_path: str | os.PathLike):
    """Return (normalized embedding, predicted gender 'M'/'F') for the largest face."""
    face = largest_face(app.get(read_image(image_path)))
    if face is None:
        raise ValueError(f"no face detected in {image_path}")
    return l2_normalize(face.embedding), face.sex


def gender_from_folder(path: Path, root: Path) -> str | None:
    for part in path.relative_to(root).parts[:-1]:
        label = GENDER_ALIASES.get(part.lower())
        if label:
            return label
    return None


class FaceIndex:
    def __init__(self, embeddings: np.ndarray, paths: Sequence[str], genders: Sequence[str]):
        embeddings = l2_normalize(np.atleast_2d(embeddings))
        if not (len(embeddings) == len(paths) == len(genders)):
            raise ValueError("embeddings, paths and genders must have the same length")
        self.embeddings = embeddings
        self.paths = np.asarray(paths, dtype=str)
        self.genders = np.asarray(genders, dtype=str)

    def __len__(self) -> int:
        return len(self.paths)

    def query(
        self,
        embedding: np.ndarray,
        gender: str | None = None,
        top_k: int = 1,
        exclude: str | None = None,
    ) -> list[Match]:
        """Top-k gallery entries by cosine similarity, optionally same-gender only."""
        sims = self.embeddings @ l2_normalize(embedding)
        mask = np.ones(len(self), dtype=bool)
        if gender:
            mask &= self.genders == gender
        if exclude:
            mask &= self.paths != str(Path(exclude).resolve())
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []
        k = min(top_k, candidates.size)
        best = candidates[np.argsort(-sims[candidates], kind="stable")[:k]]
        return [Match(str(self.paths[i]), float(sims[i]), str(self.genders[i])) for i in best]

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, embeddings=self.embeddings, paths=self.paths, genders=self.genders)

    @classmethod
    def load(cls, path: str | os.PathLike) -> FaceIndex:
        data = np.load(path)
        return cls(data["embeddings"], data["paths"].tolist(), data["genders"].tolist())

    @classmethod
    def build(cls, app, gallery: str | os.PathLike, crop_fallback: bool = True) -> FaceIndex:
        """Embed every image under ``gallery``.

        Gender comes from the folder name; images outside a recognised gender
        folder fall back to InsightFace's prediction. With ``crop_fallback``,
        images where detection fails (common for tight face crops) are resized
        and passed straight to the recogniser.
        """
        root = Path(gallery).resolve()
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise ValueError(f"no images found under {root}")

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **_):
                return x

        recogniser = app.models["recognition"]
        embeddings, paths, genders = [], [], []
        skipped = cropped = 0
        for path in tqdm(files, desc="embedding gallery", unit="img"):
            try:
                img = read_image(path)
            except ValueError:
                skipped += 1
                continue
            face = largest_face(app.get(img))
            gender = gender_from_folder(path, root)
            if face is not None:
                emb = face.embedding
                gender = gender or face.sex
            elif crop_fallback and gender:
                emb = recogniser.get_feat(img).reshape(-1)
                cropped += 1
            else:
                skipped += 1
                continue
            embeddings.append(emb)
            paths.append(str(path))
            genders.append(gender)

        print(f"indexed {len(paths)} faces ({cropped} via crop fallback, {skipped} skipped)")
        if not paths:
            raise ValueError("no faces could be embedded")
        return cls(np.stack(embeddings), paths, genders)
