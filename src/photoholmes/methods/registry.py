from enum import Enum, unique


@unique
class MethodName(Enum):
    NAIVE = "naive"
    DQ = "dq"
    SPLICEBUSTER = "splicebuster"
    CATNET = "catnet"
    EXIF_AS_LANGUAGE = "exif_as_language"
    ADAPTIVE_CFA_NET = "adaptive_cfa_net"
    NOISESNIFFER = "noisesniffer"
    PSCCNET = "psccnet"
    TRUFOR = "trufor"
    FOCAL = "focal"
