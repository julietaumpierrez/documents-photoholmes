from enum import Enum, unique


@unique
class DatasetName(Enum):
    COLUMBIA = "columbia"
    OSN = "osn"
    COVERAGE = "coverage"
    REALISTIC_TAMPERING = "realistic_tampering"
    DSO1 = "dso1"
    CASIA1_COPY_MOVE = "casia1"
    CASIA1_SPLICING = "casia1"
    AUTOSPLICE100 = "autosplice"
    AUTOSPLICE90 = "autosplice"
    AUTOSPLICE75 = "autosplice"
    TRACE_NOISE_EXO = "trace"
    TRACE_NOISE_ENDO = "trace"
    TRACE_CFA_ALG_EXO = "trace"
    TRACE_CFA_ALG_ENDO = "trace"
    TRACE_CFA_GRID_EXO = "trace"
    TRACE_CFA_GRID_ENDO = "trace"
    TRACE_JPEG_GRID_EXO = "trace"
    TRACE_JPEG_GRID_ENDO = "trace"
    TRACE_JPEG_QUALITY_EXO = "trace"
    TRACE_JPEG_QUALITY_ENDO = "trace"
    TRACE_HYBRID_EXO = "trace"
    TRACE_HYBRID_ENDO = "trace"
