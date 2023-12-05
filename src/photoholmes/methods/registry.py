from enum import Enum, unique


@unique
class MethodName(Enum):
    NAIVE = "naive"
    DQ = "dq"
    SPLICEBUSTER = "splicebuster"
    CATNET = "catnet"
    EXIF_AS_LANGUAGE = "exif_as_language"
    CFANET = "cfanet"
    NOISESNIFFER = "noisesniffer"
