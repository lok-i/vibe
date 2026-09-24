"""CPU-only contracts for the public model release (`lkrajan/vibe` on HF).

  1. the bundled manifest names only REGISTERED tasks, at the tree layout the HF
     repo mirrors (`<task-id>/checkpoint.pt`);
  2. `play --agent` offers orcs's choices UNION vibe's — the play patch stacks on
     orcs's, and a respelled Literal once dropped `release` for every task;
  3. `--agent release` ROUTES by manifest: a vibe task to vibe's, anything else to
     orcs's handler, and a named checkpoint beside it is refused;
  4. `export-agent --release` routes the same way, and the exporter imports
     without onnxruntime (only its check/viewer need it).

The downloader itself (verified, atomic, cached) is orcs's and tested there; here,
only that vibe's cache is vibe's. No network, no checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
from typing import get_args, get_type_hints

import mjlab.scripts.play as play
import orcs.release
import pytest
from mjlab.tasks.registry import list_tasks

import vibe  # noqa: F401 — register tasks, apply the play patch
from vibe import release
from vibe.core.paths import VIBE_ROOT


def test_manifest_contract():
    manifest = release.release_manifest()
    assert (manifest["repo_id"], manifest["revision"]) == ("lkrajan/vibe", "v0.1.0")
    assert set(manifest["models"]) <= set(list_tasks())
    for task_id, model in manifest["models"].items():
        ckpt = model["checkpoint"]
        assert ckpt["path"] == f"{task_id}/checkpoint.pt"
        assert ckpt["size_bytes"] > 0 and len(ckpt["sha256"]) == 64


def test_manifest_mocke_matches_lock():
    # mocke carries the obs/action contract the SONIC base is bit-coupled to: a
    # lock bump there is a re-validation, not a chore. Other rows are lineage.
    lock = json.loads((VIBE_ROOT / "deps.lock").read_text())
    compat = release.release_manifest()["compatibility"]
    assert set(compat) <= set(lock)
    assert compat["mocke"] == lock["mocke"]["sha"]


def test_play_agent_choices_are_the_union():
    hint = get_type_hints(play.PlayConfig)["agent"]
    assert {"auto", "zero", "random", "trained", "initial", "release"} <= set(get_args(hint))
    assert play.PlayConfig().agent == "auto"


class _Routed(Exception):
    pass


def _raise(tag):
    def f(task_id, **_):
        raise _Routed(tag, task_id)
    return f


@pytest.mark.parametrize("task_id, owner", [
    ("Vibe-Dodge-ImgFeat-Ext", "vibe"),
    ("Orcs-Dodge-AdaptSonic", "orcs"),
])
def test_release_routes_by_manifest(monkeypatch, task_id, owner):
    monkeypatch.setattr(release, "ensure_released_model", _raise("vibe"))
    monkeypatch.setattr(orcs.release, "ensure_released_model", _raise("orcs"))
    with pytest.raises(_Routed) as exc:
        play.run_play(task_id, play.PlayConfig(agent="release"))
    assert exc.value.args == (owner, task_id)


def test_release_refuses_a_named_checkpoint(monkeypatch):
    monkeypatch.setattr(release, "ensure_released_model", _raise("vibe"))
    cfg = play.PlayConfig(agent="release", checkpoint_file="x.pt")
    with pytest.raises(ValueError, match="--checkpoint-file"):
        play.run_play("Vibe-Dodge-ImgFeat-Ext", cfg)


def test_cache_is_vibes(tmp_path, monkeypatch):
    payload = b"released model"
    manifest = {"schema": "vibe.release.v1", "repo_id": "example/vibe", "revision": "test",
                "models": {"Vibe-Test": {"checkpoint": {
                    "path": "Vibe-Test/checkpoint.pt", "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest()}}}}
    monkeypatch.setenv("VIBE_RELEASE_ROOT", str(tmp_path))
    monkeypatch.setattr(orcs.release, "release_manifest", lambda *_: manifest)
    monkeypatch.setattr(orcs.release.urllib.request, "urlopen",
                        lambda *_a, **_k: io.BytesIO(payload))
    path = release.ensure_released_model("Vibe-Test")
    assert path == tmp_path / "test" / "Vibe-Test" / "checkpoint.pt"
    assert path.read_bytes() == payload


def _exporter(monkeypatch):
    """A fresh import of the exporter with onnxruntime unimportable."""
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    for name in ("vibe.export.agent.export_onnx", "vibe.export.agent.onnx_agent"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("vibe.export.agent.export_onnx")


def test_exporter_imports_without_onnxruntime(monkeypatch):
    ex = _exporter(monkeypatch)
    with pytest.raises(SystemExit, match="sync_deps.sh --deploy"):
        ex._onnx_agent()


@pytest.mark.parametrize("task_id, owner", [
    ("Vibe-Dodge-ImgFeat-Ext", "vibe"),
    ("Orcs-Dodge-AdaptSonic", "orcs"),
])
def test_export_release_routes_by_manifest(monkeypatch, task_id, owner):
    ex = _exporter(monkeypatch)
    monkeypatch.setattr(release, "ensure_released_model", _raise("vibe"))
    monkeypatch.setattr(orcs.release, "ensure_released_model", _raise("orcs"))
    with pytest.raises(_Routed) as exc:
        ex._resolve_checkpoint(ex.ExportConfig(release=True), task_id, "exp")
    assert exc.value.args == (owner, task_id)


def test_export_release_refuses_a_named_checkpoint(monkeypatch):
    ex = _exporter(monkeypatch)
    cfg = ex.ExportConfig(release=True, checkpoint_file="x.pt")
    with pytest.raises(ValueError, match="--release cannot be combined"):
        ex._resolve_checkpoint(cfg, "Vibe-Dodge-ImgFeat-Ext", "exp")
