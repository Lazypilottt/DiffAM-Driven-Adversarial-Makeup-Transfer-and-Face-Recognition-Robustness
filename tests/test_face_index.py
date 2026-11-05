from pathlib import Path

import numpy as np
import pytest

from makeup_attack.face_index import FaceIndex, gender_from_folder, l2_normalize


@pytest.fixture
def index():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(6, 512)).astype(np.float32)
    paths = [f"/gallery/{g}/{i}.jpg" for i, g in enumerate("MMMFFF")]
    return FaceIndex(emb, paths, list("MMMFFF")), emb


def test_embeddings_are_unit_norm(index):
    idx, _ = index
    assert np.allclose(np.linalg.norm(idx.embeddings, axis=1), 1.0, atol=1e-5)


def test_query_returns_exact_match_first(index):
    idx, emb = index
    best = idx.query(emb[4] * 3.0, top_k=1)[0]
    assert best.path.endswith("/4.jpg")
    assert best.similarity == pytest.approx(1.0, abs=1e-5)


def test_gender_filter(index):
    idx, emb = index
    matches = idx.query(emb[4], gender="M", top_k=10)
    assert len(matches) == 3
    assert all(m.gender == "M" for m in matches)


def test_results_sorted_by_similarity(index):
    idx, emb = index
    sims = [m.similarity for m in idx.query(emb[0], top_k=6)]
    assert sims == sorted(sims, reverse=True)


def test_exclude_drops_query_image(index, tmp_path):
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(3, 8))
    paths = [str((tmp_path / f"{i}.jpg").resolve()) for i in range(3)]
    idx = FaceIndex(emb, paths, ["F"] * 3)
    best = idx.query(emb[1], exclude=paths[1])[0]
    assert best.path != paths[1]


def test_empty_result_when_gender_missing(index):
    idx, emb = index
    assert idx.query(emb[0], gender="X") == []


def test_save_load_roundtrip(index, tmp_path):
    idx, emb = index
    out = tmp_path / "cache" / "gallery.npz"
    idx.save(out)
    loaded = FaceIndex.load(out)
    assert len(loaded) == len(idx)
    assert np.allclose(loaded.embeddings, idx.embeddings)
    assert loaded.query(emb[2])[0].path == idx.query(emb[2])[0].path


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        FaceIndex(np.ones((2, 4)), ["a"], ["M", "F"])


@pytest.mark.parametrize("rel, expected", [("boys/a.jpg", "M"), ("Female/x/b.png", "F"),
                                           ("women/c.jpg", "F"), ("misc/d.jpg", None)])
def test_gender_from_folder(rel, expected):
    root = Path("/pool")
    assert gender_from_folder(root / rel, root) == expected


def test_l2_normalize_handles_zero_vector():
    assert np.all(np.isfinite(l2_normalize(np.zeros(4))))


class _Face:
    def __init__(self, emb, bbox, sex="F"):
        self.embedding = np.asarray(emb, dtype=np.float32)
        self.bbox = bbox
        self.sex = sex


class _Recogniser:
    def get_feat(self, img):
        return np.full((1, 4), 0.5, dtype=np.float32)


class _FakeApp:
    """Detects one face in files named face*, nothing elsewhere."""

    models = {"recognition": _Recogniser()}

    def get(self, img):
        if img.startswith("face"):
            small = _Face([1, 0, 0, 0], (0, 0, 10, 10))
            large = _Face([0, 1, 0, 0], (0, 0, 50, 50))
            return [small, large]
        return []


def test_build_uses_largest_face_and_crop_fallback(tmp_path, monkeypatch):
    import makeup_attack.face_index as fi

    for rel in ["boys/face1.jpg", "girls/crop1.png", "misc/crop2.jpg", "misc/face2.jpg", "notes.txt"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    monkeypatch.setattr(fi, "read_image", lambda p: Path(p).name)

    idx = fi.FaceIndex.build(_FakeApp(), tmp_path)
    by_name = {Path(p).name: (g, e) for p, g, e in zip(idx.paths, idx.genders, idx.embeddings)}

    assert set(by_name) == {"face1.jpg", "crop1.png", "face2.jpg"}  # misc/crop2 has no gender label
    assert by_name["face1.jpg"][0] == "M"
    assert np.allclose(by_name["face1.jpg"][1], [0, 1, 0, 0])  # largest face wins
    assert by_name["crop1.png"][0] == "F"
    assert by_name["face2.jpg"][0] == "F"  # predicted sex when folder has no label
