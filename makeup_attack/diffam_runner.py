"""Build and launch single-image DiffAM edits.

DiffAM is used as an external checkout (see ``scripts/setup_diffam.sh``) and
driven through its own ``main.py``, so the upstream code stays untouched.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Sampling settings from the DiffAM single-image commands that were run in
# notebooks/diffam_colab_experiments.ipynb.
PRESETS = {
    "MT": {"flag": "--edit_one_image_MT", "config": "celeba.yml",
           "t_0": 60, "n_inv_step": 20, "n_train_step": 6},
    "MR": {"flag": "--edit_one_image_MR", "config": "MT.yml",
           "t_0": 300, "n_inv_step": 40, "n_train_step": 6},
}


@dataclass
class DiffAMJob:
    diffam_dir: Path
    image: Path
    checkpoint: Path
    mode: str = "MT"
    steps: int = 6
    exp_dir: str = "./runs/similarity_guided"

    def command(self) -> list[str]:
        if self.mode not in PRESETS:
            raise ValueError(f"mode must be one of {sorted(PRESETS)}")
        p = PRESETS[self.mode]
        steps = max(1, min(int(self.steps), p["t_0"]))
        return [
            sys.executable, "main.py", p["flag"],
            "--config", p["config"],
            "--exp", self.exp_dir,
            "--n_iter", "1",
            "--t_0", str(p["t_0"]),
            "--n_inv_step", str(p["n_inv_step"]),
            "--n_train_step", str(p["n_train_step"]),
            "--n_test_step", str(steps),
            "--img_path", str(Path(self.image).resolve()),
            "--model_path", str(Path(self.checkpoint).resolve()),
        ]

    def run(self, dry_run: bool = False) -> int:
        cmd = self.command()
        print(f"[diffam] cd {shlex.quote(str(self.diffam_dir))} && {shlex.join(cmd)}")
        if dry_run:
            return 0
        if not (Path(self.diffam_dir) / "main.py").is_file():
            raise FileNotFoundError(
                f"{self.diffam_dir}/main.py not found; run scripts/setup_diffam.sh first"
            )
        return subprocess.run(cmd, cwd=self.diffam_dir).returncode
