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
