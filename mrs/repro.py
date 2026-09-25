"""Reproducibility metadata for training and evaluation runs."""

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(args, cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def environment_metadata(project_dir: Optional[Path] = None) -> Dict[str, Any]:
    project_dir = Path(project_dir or Path.cwd()).resolve()
    metadata: Dict[str, Any] = {
        "git_commit": _git(["rev-parse", "HEAD"], project_dir),
        "git_branch": _git(["branch", "--show-current"], project_dir),
        "git_dirty": bool(_git(["status", "--porcelain"], project_dir)),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import torch

        metadata.update({
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        })
    except ImportError:
        metadata.update({
            "torch": None,
            "cuda_runtime": None,
            "cudnn": None,
            "cuda_available": False,
            "gpu": [],
        })
    return metadata


@dataclass
class RunMetadata:
    run_id: str
    method: str
    config: Dict[str, Any]
    data_seed: int
    model_seed: int
    started_at: str
    finished_at: Optional[str]
    best_validation_checkpoint: Optional[str]
    status: str
    environment: Dict[str, Any]


class RunRecorder:
    """Atomically maintain the required metadata alongside a run."""

    def __init__(
        self,
        output_dir: Path,
        method: str,
        config: Dict[str, Any],
        data_seed: int,
        model_seed: int,
        project_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "run_metadata.json"
        self.metadata = RunMetadata(
            run_id=run_id or self.output_dir.name,
            method=method,
            config=config,
            data_seed=int(data_seed),
            model_seed=int(model_seed),
            started_at=utc_now(),
            finished_at=None,
            best_validation_checkpoint=None,
            status="running",
            environment=environment_metadata(project_dir),
        )
        self._write()

    @classmethod
    def resume(cls, output_dir: Path) -> "RunRecorder":
        output_dir = Path(output_dir)
        path = output_dir / "run_metadata.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "running":
            raise RuntimeError("only an interrupted running record can be resumed")
        recorder = cls.__new__(cls)
        recorder.output_dir = output_dir
        recorder.path = path
        recorder.metadata = RunMetadata(**payload)
        return recorder

    def finish(self, status: str, best_validation_checkpoint: Optional[Path] = None) -> None:
        if status not in {"completed", "failed", "timed_out"}:
            raise ValueError(f"unsupported terminal status: {status}")
        self.metadata.status = status
        self.metadata.finished_at = utc_now()
        self.metadata.best_validation_checkpoint = (
            str(best_validation_checkpoint) if best_validation_checkpoint else None
        )
        self._write()

    def _write(self) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(self.metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
