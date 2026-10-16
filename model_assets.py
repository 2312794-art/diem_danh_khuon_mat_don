"""Download and verify the shared anti-spoof ONNX model on demand."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


ANTISPOOF_MODEL_URL = (
    "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/"
    "MiniFASNetV2.onnx"
)
ANTISPOOF_MODEL_SHA256 = "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_antispoof_model(model_path: Path) -> Path:
    """Return a checksum-verified model, downloading it atomically if needed."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.is_file() and _sha256(model_path) == ANTISPOOF_MODEL_SHA256:
        return model_path

    temp_path = None
    try:
        request = Request(
            ANTISPOOF_MODEL_URL,
            headers={"User-Agent": "FaceAttendance/1.0", "Accept": "application/octet-stream"},
        )
        with urlopen(request, timeout=30) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix="antispoof_", suffix=".download",
                dir=model_path.parent, delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                digest = hashlib.sha256()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temp_file.write(chunk)
                    digest.update(chunk)
                temp_file.flush()
                os.fsync(temp_file.fileno())

        if digest.hexdigest() != ANTISPOOF_MODEL_SHA256:
            raise ValueError("SHA-256 của model MiniFASNetV2 không khớp")
        os.replace(temp_path, model_path)
        temp_path = None
        return model_path
    except Exception as exc:
        raise RuntimeError(f"Không thể tải model anti-spoof vào {model_path}: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
