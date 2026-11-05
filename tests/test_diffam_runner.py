from pathlib import Path

import pytest

from makeup_attack.diffam_runner import DiffAMJob


def _job(**kw):
    defaults = dict(diffam_dir=Path("third_party/DiffAM"), image=Path("face.jpg"),
                    checkpoint=Path("ckpt.pth"))
    defaults.update(kw)
    return DiffAMJob(**defaults)


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_transfer_preset():
    cmd = _job(mode="MT", steps=12).command()
    assert "--edit_one_image_MT" in cmd
    assert _arg(cmd, "--config") == "celeba.yml"
    assert _arg(cmd, "--t_0") == "60"
    assert _arg(cmd, "--n_test_step") == "12"


def test_removal_preset():
    cmd = _job(mode="MR", steps=40).command()
    assert "--edit_one_image_MR" in cmd
    assert _arg(cmd, "--config") == "MT.yml"
    assert _arg(cmd, "--t_0") == "300"


def test_steps_clamped_to_t0():
    cmd = _job(mode="MT", steps=500).command()
    assert _arg(cmd, "--n_test_step") == "60"


def test_paths_are_absolute():
    cmd = _job().command()
    assert Path(_arg(cmd, "--img_path")).is_absolute()
    assert Path(_arg(cmd, "--model_path")).is_absolute()


def test_unknown_mode():
    with pytest.raises(ValueError):
        _job(mode="XX").command()


def test_missing_checkout_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _job(diffam_dir=tmp_path).run()


def test_dry_run_does_not_need_checkout(tmp_path):
    assert _job(diffam_dir=tmp_path).run(dry_run=True) == 0
