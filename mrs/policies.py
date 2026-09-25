"""Single registry for current learned policies and checkpoint dispatch."""

from dataclasses import replace

from .decentralized_am import DecentralizedAMConfig, DecentralizedAMPolicy
from .decentralized_policy import HetMRTAPolicyConfig, HetMRTAPolicy
from .formal_medp import FormalMEDPConfig, FormalMEDPPolicy
from .decentralized_am_checkpoint import load_decentralized_am_checkpoint, save_decentralized_am_checkpoint
from .decentralized_checkpoint import load_hetmrta_checkpoint, save_hetmrta_checkpoint
from .formal_medp_checkpoint import load_formal_medp_checkpoint, save_formal_medp_checkpoint
from .method_names import LEARNED_METHOD_LABELS as METHOD_LABELS

FORMAL_MEDP_METHODS = ("medp_formal", "medp_1r")
STUDY_METHODS = ("hetmrta_mrs", "medp_1r", "d_am")


def build_model(method, device, medp_ablation="full", medp_forward_chunk_size=None,
                *, medp_decoder_glimpses=2, medp_query_fusion_contexts=4,
                n_mbr=4, n_dor=8):
    if method not in FORMAL_MEDP_METHODS and medp_ablation != "full":
        raise ValueError("MEDP ablations only apply to formal MEDP methods")
    if method == "hetmrta_mrs":
        return HetMRTAPolicy(HetMRTAPolicyConfig(n_mbr=n_mbr, n_dor=n_dor)).to(device)
    if method == "d_am":
        return DecentralizedAMPolicy(DecentralizedAMConfig(n_mbr=n_mbr, n_dor=n_dor)).to(device)
    if method in FORMAL_MEDP_METHODS:
        config = FormalMEDPConfig(backbone=DecentralizedAMConfig(n_mbr=n_mbr, n_dor=n_dor),
                                 ablation=medp_ablation,
                                 decoder_glimpses=medp_decoder_glimpses,
                                 query_fusion_contexts=medp_query_fusion_contexts)
        if method == "medp_1r":
            config = replace(config, recurrent_steps=1)
        if medp_forward_chunk_size is not None:
            config = replace(config, forward_chunk_size=int(medp_forward_chunk_size))
        return FormalMEDPPolicy(config).to(device)
    raise ValueError(f"unsupported decentralized method {method!r}")


def _checkpoint_functions(method):
    if method == "hetmrta_mrs":
        return load_hetmrta_checkpoint, save_hetmrta_checkpoint
    if method == "d_am":
        return load_decentralized_am_checkpoint, save_decentralized_am_checkpoint
    if method in FORMAL_MEDP_METHODS:
        return load_formal_medp_checkpoint, save_formal_medp_checkpoint
    raise ValueError(f"unsupported decentralized method {method!r}")


def load_policy_checkpoint(method, path, device):
    return _checkpoint_functions(method)[0](path, device)


def save_policy_checkpoint(method, path, model, optimizer, **metadata):
    return _checkpoint_functions(method)[1](path, model, optimizer, **metadata)
