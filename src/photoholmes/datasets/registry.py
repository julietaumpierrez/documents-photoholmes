from enum import Enum, unique


@unique
class DatasetName(Enum):
    COLUMBIA = "columbia"
    OSN = "osn"
    COVERAGE = "coverage"
    REALISTIC_TAMPERING = "realistic_tampering"
    DSO1 = "dso1"
