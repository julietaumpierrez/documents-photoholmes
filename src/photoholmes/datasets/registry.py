from enum import Enum, unique


@unique
class DatasetName(Enum):
    COLUMBIA = "columbia"
    OSN = "osn"
    COVERAGE = "coverage"
    REALISTIC_TAMPERING = "realistic_tampering"
    DSO1 = "dso1"
    CASIA1_COPY_MOVE = "casia1_copy_move"
    CASIA1_SPLICING = "casia1_splicing"
    AUTOSPLICE100 = "autosplice_100"
    AUTOSPLICE90 = "autosplice_90"
    AUTOSPLICE75 = "autosplice_75"
    TRACE_NOISE_EXO = "trace_noise_exo"
    TRACE_NOISE_ENDO = "trace_noise_endo"
    TRACE_CFA_ALG_EXO = "trace_cfa_alg_exo"
    TRACE_CFA_ALG_ENDO = "trace_cfa_alg_endo"
    TRACE_CFA_GRID_EXO = "trace_cfa_grid_exo"
    TRACE_CFA_GRID_ENDO = "trace_cfa_grid_endo"
    TRACE_JPEG_GRID_EXO = "trace_jpeg_grid_exo"
    TRACE_JPEG_GRID_ENDO = "trace_jpeg_grid_endo"
    TRACE_JPEG_QUALITY_EXO = "trace_jpeg_quality_exo"
    TRACE_JPEG_QUALITY_ENDO = "trace_jpeg_quality_endo"
    TRACE_HYBRID_EXO = "trace_hybrid_exo"
    TRACE_HYBRID_ENDO = "trace_hybrid_endo"
