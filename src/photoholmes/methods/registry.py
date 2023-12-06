from enum import Enum, unique


@unique
class MethodName(Enum):
    NAIVE = "naive"
    DQ = "dq"
    SPLICEBUSTER = "splicebuster"
    CATNET = "catnet"
    CFANET = "cfanet"
    PSCCNET = "psccnet"
