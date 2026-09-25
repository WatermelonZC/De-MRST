"""Load and verify immutable formal experiment protocol bundles."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_PROTOCOL_ID = "v1"


def source_file_hash(path: Path) -> str:
    """Portable text-source hash: canonicalize Windows/Unix line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_protocol_source_files(project_root: Path, protocol: Dict[str, Any]) -> None:
    for relative, expected in protocol.get("source_sha256_lf", {}).items():
        path = (Path(project_root) / relative).resolve()
        root = Path(project_root).resolve()
        if root not in path.parents:
            raise ValueError("protocol source path is outside the workspace")
        if not path.is_file() or source_file_hash(path) != expected:
            raise RuntimeError(f"frozen protocol source mismatch: {relative}")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FrozenProtocol:
    protocol: Dict[str, Any]
    training_validation_seeds: Dict[str, Any]
    test_seeds: Dict[str, Any]
    canonical_sha256: str
    protocol_id: str = DEFAULT_PROTOCOL_ID


def _validate_protocol_id(protocol_id: str) -> str:
    protocol_id = str(protocol_id)
    if not protocol_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in protocol_id
    ):
        raise ValueError("protocol id must contain only lowercase letters, digits, and underscores")
    return protocol_id


def load_frozen_protocol(
    project_root: Path,
    protocol_id: str = DEFAULT_PROTOCOL_ID,
) -> FrozenProtocol:
    project_root = Path(project_root).resolve()
    protocol_id = _validate_protocol_id(protocol_id)
    config = project_root / "configs"
    protocol = json.loads(
        (config / f"experiment_protocol_{protocol_id}.json").read_text(encoding="utf-8")
    )
    training = json.loads(
        (config / f"training_validation_seeds_{protocol_id}.json").read_text(
            encoding="utf-8"
        )
    )
    test = json.loads(
        (config / f"test_seeds_{protocol_id}.json").read_text(encoding="utf-8")
    )
    expected_sha = (
        (config / f"experiment_protocol_{protocol_id}.sha256")
        .read_text(encoding="ascii")
        .split()[0]
    )
    actual_sha = canonical_hash(protocol)
    if actual_sha != expected_sha:
        raise RuntimeError("formal experiment protocol canonical hash mismatch")
    expected_manifests = protocol["seed_manifest_sha256"]
    if canonical_hash(training) != expected_manifests["training_validation"]:
        raise RuntimeError("training/validation seed manifest hash mismatch")
    if canonical_hash(test) != expected_manifests["test"]:
        raise RuntimeError("test seed manifest hash mismatch")
    if protocol.get("status") != "frozen_before_formal_training_and_test_evaluation":
        raise RuntimeError("experiment protocol is not in the frozen state")
    return FrozenProtocol(protocol, training, test, actual_sha, protocol_id)


def setting_identifier(setting: Dict[str, Any]) -> str:
    return "n{task_count}_m{n_mbr}_o{n_dor}".format(**setting)


def training_settings(frozen: FrozenProtocol) -> List[Dict[str, Any]]:
    manifest = frozen.training_validation_seeds
    settings = manifest.get("training_and_validation_settings")
    if settings is None:
        settings = [manifest["training_and_validation_setting"]]
    normalized = []
    for item in settings:
        setting = dict(item)
        setting.setdefault("setting_id", setting_identifier(setting))
        normalized.append(setting)
    identifiers = [setting["setting_id"] for setting in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("training setting identifiers are not unique")
    return normalized


def learned_model_seeds(frozen: FrozenProtocol) -> List[int]:
    test = frozen.protocol["test"]
    if "learned_model_seeds" in test:
        return list(test["learned_model_seeds"])
    return list(test["learned_and_stochastic_method_seeds"])


def protocol_objective_mode(frozen: FrozenProtocol) -> str:
    """Return the objective mode frozen into a protocol, defaulting to v4.

    Historical protocol bundles predate the explicit objective contract, so
    they retain their original v4 behavior. New bundles must declare the mode
    in the immutable objective section.
    """
    objective = frozen.protocol.get("objective", {})
    if not isinstance(objective, dict):
        raise RuntimeError("protocol objective must be an object")
    from .core import normalize_objective_mode

    return normalize_objective_mode(objective.get("mode", "v4"))


def stochastic_search_seeds(frozen: FrozenProtocol) -> List[int]:
    test = frozen.protocol["test"]
    if "stochastic_search_seeds" in test:
        return list(test["stochastic_search_seeds"])
    return list(test["learned_and_stochastic_method_seeds"])


def uses_specialized_checkpoint_layout(frozen: FrozenProtocol) -> bool:
    return "training_and_validation_settings" in frozen.training_validation_seeds


def select_training_setting(
    frozen: FrozenProtocol,
    setting_id: Optional[str] = None,
) -> Dict[str, Any]:
    settings = training_settings(frozen)
    if setting_id is None:
        if len(settings) != 1:
            raise ValueError("--setting-id is required for a multi-setting protocol")
        return settings[0]
    matches = [setting for setting in settings if setting["setting_id"] == setting_id]
    if not matches:
        choices = ", ".join(setting["setting_id"] for setting in settings)
        raise ValueError(f"unknown training setting {setting_id!r}; choose one of: {choices}")
    return matches[0]


def write_frozen_bundles(bundles, project_root: Path, *, verify_only=False):
    """Validate every artifact before exporting any; never replace a frozen file."""
    directory = Path(project_root) / "configs"
    bundles = tuple(bundles)
    exports = []
    for pid, payloads in bundles:
        _validate_protocol_id(pid)
        names = {f"experiment_protocol_{pid}.json", f"training_validation_seeds_{pid}.json",
                 f"test_seeds_{pid}.json"}
        if set(payloads) != names:
            raise ValueError("a protocol bundle must contain exactly its three named JSON artifacts")
        protocol = payloads[f"experiment_protocol_{pid}.json"]
        for key, prefix in (("training_validation", "training_validation_seeds"), ("test", "test_seeds")):
            if canonical_hash(payloads[f"{prefix}_{pid}.json"]) != protocol["seed_manifest_sha256"][key]:
                raise ValueError("seed manifest hash differs from protocol")
        for name, value in payloads.items():
            path = directory / name
            if verify_only and not path.is_file():
                raise FileNotFoundError(path)
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
                raise FileExistsError(f"refusing to overwrite frozen artifact {path}")
            exports.append((path, json.dumps(value, indent=2, sort_keys=True) + "\n"))
        digest = canonical_hash(protocol)
        sha = directory / f"experiment_protocol_{pid}.sha256"
        if verify_only and not sha.is_file():
            raise FileNotFoundError(sha)
        if sha.exists() and sha.read_text(encoding="ascii").split()[0] != digest:
            raise FileExistsError(f"refusing to overwrite frozen hash {sha}")
        exports.append((sha, f"{digest}  experiment_protocol_{pid}.canonical-json\n"))
    if not verify_only:
        directory.mkdir(parents=True, exist_ok=True)
        for path, content in exports:
            if not path.exists():
                with path.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
    return [{"protocol_id": pid, "sha256": canonical_hash(payloads[f"experiment_protocol_{pid}.json"])}
            for pid, payloads in bundles]
