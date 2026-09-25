"""Public experiment names and directories; legacy CLI IDs remain stable."""

DE_MRST_NAME = "De-MRST"
LEARNED_METHOD_LABELS = {
    "hetmrta_mrs": "HetMRTA-RL-MRS",
    "d_am": "D-AM",
    "medp_1r": DE_MRST_NAME,
    "medp_formal": DE_MRST_NAME,
}
METHOD_RUN_DIRECTORIES = {
    "medp_1r": DE_MRST_NAME,
    "medp_formal": "De-MRST-development",
}


def method_directory(method):
    return METHOD_RUN_DIRECTORIES.get(method, method)
