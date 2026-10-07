"""Calibrate safetensors checkpoints without instantiating a model."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import tempfile
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .core import _calibrate_weight, _validate_parameters

_LOG = logging.getLogger(__name__)
_EMBED = "model.embed_tokens.weight"
_HEAD = "lm_head.weight"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.exists() or str(value).startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"Checkpoint directory not found: {value}")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(value), allow_patterns=[
        "*.safetensors", "*.json", "*.model", "*.txt", "*.jinja", "*.tiktoken",
    ]))


def _read_json(path: Path):
    return json.loads(path.read_text())


def _safe_member(root: Path, name: str) -> Path:
    member = Path(name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Invalid checkpoint member: {name}")
    return root / member


def _inventory(root: Path):
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {root}")
    config = _read_json(config_path)
    if config.get("quantization_config") is not None or (root / "adapter_config.json").exists():
        raise ValueError("Use a full, non-quantized checkpoint; adapters and quantization are unsupported")
    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = _read_json(index)["weight_map"]
    elif (root / "model.safetensors").exists():
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as f:
            mapping = dict.fromkeys(f.keys(), "model.safetensors")
    else:
        raise FileNotFoundError(f"No model.safetensors or safetensors index in {root}")
    if not mapping:
        raise ValueError("Checkpoint contains no tensors")
    shapes = {}
    for filename in sorted(set(mapping.values())):
        if not filename.endswith(".safetensors"):
            raise ValueError(f"Unsupported weights file: {filename}")
        with safe_open(_safe_member(root, filename), framework="pt", device="cpu") as f:
            expected = {k for k, v in mapping.items() if v == filename}
            if set(f.keys()) != expected:
                raise ValueError(f"Index and shard disagree: {filename}")
            for name in f.keys():
                shapes[name] = tuple(f.get_slice(name).get_shape())
    return config, mapping, shapes


def _load(root, mapping, name):
    with safe_open(_safe_member(root, mapping[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _canonical(name, tied):
    return _EMBED if tied and name == _HEAD else name


def _names(shapes, tied):
    result = {_canonical(k, tied): k for k in sorted(shapes)}
    if tied and _EMBED in shapes:
        result[_EMBED] = _EMBED
    return result


def _check_tied(root, mapping, shapes, tied):
    if tied and _EMBED in shapes and _HEAD in shapes:
        if not torch.equal(_load(root, mapping, _EMBED), _load(root, mapping, _HEAD)):
            raise ValueError(f"config ties embeddings, but stored embedding/head disagree: {root}")


def _copy_metadata(source, output):
    for path in source.iterdir():
        if path.is_file() and (
            path.name in {"config.json", "generation_config.json", "special_tokens_map.json", "added_tokens.json", "merges.txt"}
            or path.name.startswith(("tokenizer", "vocab", "chat_template"))
            or path.suffix in {".model", ".tiktoken", ".jinja"}
        ):
            shutil.copy2(path, output / path.name)
    if (source / "chat_templates").is_dir():
        shutil.copytree(source / "chat_templates", output / "chat_templates")


@torch.no_grad()
def space_calibrate(
    pre_sft: str | Path,
    post_sft: str | Path,
    output_dir: str | Path,
    *,
    rho: float = 0.5,
    alpha: float = 1.0,
    device: str = "auto",
) -> Path:
    """Write a calibrated checkpoint and return its absolute directory.

    Inputs may be local HF directories or Hub IDs. Output must not exist.
    All floating 2D weights are calibrated; other tensors are copied from SFT.
    Computation holds one tensor pair plus SVD workspace and a CPU output shard.
    Shared embedding/head tensors are transformed once and reused.
    """
    _validate_parameters(rho, alpha)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    base, post = _resolve(pre_sft), _resolve(post_sft)
    for source in (base, post):
        if destination == source or source in destination.parents:
            raise ValueError("Output must be outside both input checkpoint directories")
    config0, map0, shapes0 = _inventory(base)
    config1, map1, shapes1 = _inventory(post)
    tied = bool(config1.get("tie_word_embeddings", False))
    if bool(config0.get("tie_word_embeddings", False)) != tied:
        raise ValueError("Input checkpoints disagree on tie_word_embeddings")
    if config0.get("model_type") != config1.get("model_type"):
        raise ValueError("Input checkpoints have different model_type")
    names0, names1 = _names(shapes0, tied), _names(shapes1, tied)
    if names0.keys() != names1.keys():
        missing = sorted(names1.keys() - names0.keys())[:5]
        extra = sorted(names0.keys() - names1.keys())[:5]
        raise ValueError(f"Input tensor names do not match; missing in pre={missing}, extra in pre={extra}")
    for name in names1:
        if shapes0[names0[name]] != shapes1[names1[name]]:
            raise ValueError(f"Input tensor shapes differ: {name}")
    _check_tied(base, map0, shapes0, tied)
    _check_tied(post, map1, shapes1, tied)
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if compute_device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu and cuda[:index]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    started = time.monotonic()
    layers = {}
    tied_result = None
    try:
        _copy_metadata(post, staging)
        for filename in sorted(set(map1.values())):
            tensors = {}
            with safe_open(_safe_member(post, filename), framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for name in f.keys():
                    canonical = _canonical(name, tied)
                    current = f.get_tensor(name)
                    if current.ndim != 2:
                        tensors[name] = current
                        continue
                    if tied and canonical == _EMBED and tied_result is not None:
                        tensors[name] = tied_result
                        continue
                    before = _load(base, map0, names0[canonical])
                    calibrated, stats = _calibrate_weight(before, current.to(compute_device), rho=rho, alpha=alpha)
                    calibrated = calibrated.cpu()
                    del before
                    tensors[name] = calibrated
                    layers[canonical] = {"shape": list(current.shape), "dtype": str(current.dtype), **stats}
                    _LOG.info("%s: k=%s", canonical, stats["selected_rank"])
                    if tied and canonical == _EMBED:
                        tied_result = calibrated
            output_shard = _safe_member(staging, filename)
            output_shard.parent.mkdir(parents=True, exist_ok=True)
            # save_file disallows aliasing two entries; duplicate only at serialization.
            if tied and _EMBED in tensors and _HEAD in tensors:
                tensors[_HEAD] = tensors[_HEAD].clone()
            save_file(tensors, output_shard, metadata=metadata)
            del tensors
        if (post / "model.safetensors.index.json").exists():
            shutil.copy2(post / "model.safetensors.index.json", staging / "model.safetensors.index.json")
        report = {
            "version": "0.1.0", "pre_sft": str(pre_sft), "post_sft": str(post_sft),
            "resolved_pre_sft": str(base), "resolved_post_sft": str(post),
            "rho": rho, "alpha": alpha, "rank_rule": "squared_singular_value_energy",
            "target_modules": "all_2d", "device": str(compute_device),
            "num_calibrated_matrices": len(layers), "layers": layers,
            "elapsed_seconds": time.monotonic() - started,
        }
        (staging / "space_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        _inventory(staging)
        if destination.exists():
            raise FileExistsError(f"Output appeared during calibration: {destination}")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
