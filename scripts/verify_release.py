"""Verify paper protocols and the published De-MRST checkpoints."""

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mrs.de_mrst_checkpoint import load_de_mrst_checkpoint
from mrs.policies import build_model
from mrs.protocol import load_frozen_protocol, validate_protocol_source_files


def main():
    manifest = json.loads((ROOT / "checkpoints" / "manifest.json").read_text(encoding="utf-8"))
    for count in (10, 20, 50, 100):
        frozen = load_frozen_protocol(ROOT, f"paper_n{count}")
        validate_protocol_source_files(ROOT, frozen.protocol)
        expected = frozen.protocol["method_architectures"]["de_mrst"]
        model = build_model(
            "de_mrst", "cpu", de_mrst_decoder_glimpses=1,
            de_mrst_query_fusion_contexts=3,
            de_mrst_forward_chunk_size=frozen.protocol["decentralized_training"]["de_mrst_policy_forward_chunk_size"],
        )
        assert model.de_mrst_config.to_dict() == expected["policy_config"]
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected["parameter_count"]

        path = ROOT / "checkpoints" / f"n{count}" / "best.pt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest[f"n{count}"]["sha256"], path
        _, payload = load_de_mrst_checkpoint(path)
        assert payload["training_config"]["protocol_id"] == manifest[f"n{count}"]["training_protocol_id"]
        print(f"n{count}: protocol and checkpoint OK")


if __name__ == "__main__":
    main()
