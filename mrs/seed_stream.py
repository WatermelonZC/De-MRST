"""Deterministic compact seed streams for large frozen training budgets."""

import math
from typing import Dict, List


def validate_seed_stream(specification: Dict[str, int]) -> None:
    required = ("algorithm", "count", "lower_bound", "modulus", "multiplier", "increment")
    missing = [key for key in required if key not in specification]
    if missing:
        raise ValueError("seed stream is missing: " + ", ".join(missing))
    if specification["algorithm"] != "affine_permutation_v1":
        raise ValueError("unsupported seed stream algorithm")
    count = int(specification["count"])
    modulus = int(specification["modulus"])
    multiplier = int(specification["multiplier"])
    if count <= 0 or modulus <= 0 or count > modulus:
        raise ValueError("invalid seed stream count or modulus")
    if math.gcd(multiplier, modulus) != 1:
        raise ValueError("seed stream multiplier must be coprime to modulus")


def seed_at(specification: Dict[str, int], index: int) -> int:
    validate_seed_stream(specification)
    index = int(index)
    count = int(specification["count"])
    if not 0 <= index < count:
        raise IndexError("seed stream index is out of range")
    return int(specification["lower_bound"]) + (
        int(specification["multiplier"]) * index
        + int(specification["increment"])
    ) % int(specification["modulus"])


def seed_slice(
    specification: Dict[str, int], start: int, stop: int
) -> List[int]:
    validate_seed_stream(specification)
    start, stop = int(start), int(stop)
    count = int(specification["count"])
    if not 0 <= start <= stop <= count:
        raise IndexError("seed stream slice is out of range")
    lower = int(specification["lower_bound"])
    multiplier = int(specification["multiplier"])
    increment = int(specification["increment"])
    modulus = int(specification["modulus"])
    return [
        lower + (multiplier * index + increment) % modulus
        for index in range(start, stop)
    ]


def training_seed_count(manifest: Dict[str, object]) -> int:
    if "training_seed_stream" in manifest:
        specification = manifest["training_seed_stream"]
        validate_seed_stream(specification)
        return int(specification["count"])
    return len(manifest["training_instance_seeds"])


def training_seed_slice(
    manifest: Dict[str, object], start: int, stop: int
) -> List[int]:
    if "training_seed_stream" in manifest:
        return seed_slice(manifest["training_seed_stream"], start, stop)
    seeds = manifest["training_instance_seeds"]
    if not 0 <= start <= stop <= len(seeds):
        raise IndexError("training seed slice is out of range")
    return list(seeds[start:stop])
