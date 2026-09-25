"""Single registry for current learned policies and checkpoint dispatch."""

from dataclasses import replace

from .decentralized_am import DecentralizedAMConfig, DecentralizedAMPolicy
from .decentralized_policy import HetMRTAPolicyConfig, HetMRTAPolicy
from .de_mrst import DeMRSTConfig, DeMRSTPolicy
from .decentralized_am_checkpoint import load_decentralized_am_checkpoint, save_decentralized_am_checkpoint
from .decentralized_checkpoint import load_hetmrta_checkpoint, save_hetmrta_checkpoint
from .de_mrst_checkpoint import load_de_mrst_checkpoint, save_de_mrst_checkpoint
from .method_names import LEARNED_METHOD_LABELS as METHOD_LABELS

DE_MRST_METHODS = ("de_mrst",)


def build_model(method, device, de_mrst_ablation="full", de_mrst_forward_chunk_size=None,
                *, de_mrst_decoder_glimpses=1, de_mrst_query_fusion_contexts=3,
                n_mbr=4, n_dor=8):
    if method not in DE_MRST_METHODS and de_mrst_ablation != "full":
        raise ValueError("De-MRST ablations only apply to De-MRST")
    if method == "hetmrta_mrs":
        return HetMRTAPolicy(HetMRTAPolicyConfig(n_mbr=n_mbr, n_dor=n_dor)).to(device)
    if method == "d_am":
        return DecentralizedAMPolicy(DecentralizedAMConfig(n_mbr=n_mbr, n_dor=n_dor)).to(device)
    if method in DE_MRST_METHODS:
        config = DeMRSTConfig(
            backbone=DecentralizedAMConfig(n_mbr=n_mbr, n_dor=n_dor),
            ablation=de_mrst_ablation,
            decoder_glimpses=de_mrst_decoder_glimpses,
            query_fusion_contexts=de_mrst_query_fusion_contexts,
        )
        if de_mrst_forward_chunk_size is not None:
            config = replace(config, forward_chunk_size=int(de_mrst_forward_chunk_size))
        return DeMRSTPolicy(config).to(device)
    raise ValueError(f"unsupported decentralized method {method!r}")


def _checkpoint_functions(method):
    if method == "hetmrta_mrs":
        return load_hetmrta_checkpoint, save_hetmrta_checkpoint
    if method == "d_am":
        return load_decentralized_am_checkpoint, save_decentralized_am_checkpoint
    if method in DE_MRST_METHODS:
        return load_de_mrst_checkpoint, save_de_mrst_checkpoint
    raise ValueError(f"unsupported decentralized method {method!r}")


def load_policy_checkpoint(method, path, device):
    return _checkpoint_functions(method)[0](path, device)


def save_policy_checkpoint(method, path, model, optimizer, **metadata):
    return _checkpoint_functions(method)[1](path, model, optimizer, **metadata)
