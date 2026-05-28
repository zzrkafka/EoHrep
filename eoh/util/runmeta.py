"""Collect environment + git + LLM identity metadata at run start.

Goal: any artifact produced today must be re-traceable to (a) the source
revision that produced it, (b) the Python/numpy versions in use, (c) the
exact LLM weights served. None of this is in the algorithm — it's the
forensics layer.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from typing import Any

import requests

logger = logging.getLogger("eoh")


def _git_info(cwd: str | None = None) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        if commit.returncode != 0:
            return {"available": False}
        return {
            "available": True,
            "commit": commit.stdout.strip(),
            "branch": branch.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {"available": False}


def _gpu_info() -> dict[str, Any]:
    """Query nvidia-smi if available; return {} on any failure."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return {}
        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "name": parts[0],
                    "memory_total_mib": int(parts[1]),
                    "memory_free_mib": int(parts[2]),
                    "driver_version": parts[3],
                })
        return {"nvidia": gpus}
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
        return {}


def _numpy_version() -> str:
    try:
        import numpy as np
        return np.__version__
    except ImportError:
        return "missing"


def _ollama_meta(base_url: str, model: str) -> dict[str, Any]:
    """Query Ollama for server version and the model's exact digest."""
    info: dict[str, Any] = {"available": False, "base_url": base_url, "model": model}
    try:
        v = requests.get(f"{base_url.rstrip('/')}/api/version", timeout=5)
        if v.status_code == 200:
            info["available"] = True
            info["server_version"] = v.json().get("version", "unknown")
    except Exception as e:
        info["error_version"] = str(e)
        return info

    try:
        # Digest lives in /api/tags, not /api/show.
        t = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        if t.status_code == 200:
            for m in t.json().get("models", []):
                if m.get("name") == model:
                    info["digest"] = m.get("digest", "unknown")
                    info["size_bytes"] = m.get("size")
                    info["modified_at"] = m.get("modified_at")
                    details = m.get("details", {}) or {}
                    info["parameter_size"] = details.get("parameter_size")
                    info["quantization_level"] = details.get("quantization_level")
                    info["family"] = details.get("family")
                    break
            else:
                info["error_tags"] = f"Model '{model}' not in /api/tags"
        else:
            info["error_tags"] = f"HTTP {t.status_code}"
    except Exception as e:
        info["error_tags"] = str(e)
    return info


def collect_env_metadata(
    cwd: str | None = None,
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
) -> dict[str, Any]:
    """Build the env metadata dict written to runs/<ts>/env.json."""
    meta: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "node": platform.node(),
        },
        "numpy_version": _numpy_version(),
        "git": _git_info(cwd),
        "gpu": _gpu_info(),
    }
    if ollama_base_url and ollama_model:
        meta["ollama"] = _ollama_meta(ollama_base_url, ollama_model)
    return meta


def dump_env(meta: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
