from enum import Enum, unique

from .base import Method
from .naive.method import Naive

@unique
class MethodType(Enum):
    NAIVE = 0

def string_to_type(method_name:str) -> MethodType:
    if method_name=="naive":
        return MethodType.NAIVE

class MethodFactory:
    @staticmethod
    def create(method_name:str, config:dict = None) -> Method:
        '''Instantiates method corresponding to the name passed, from config
        '''
        method_type = string_to_type(method_name)
        if method_type == MethodType.NAIVE:
            return Naive.from_config(config)
        else:
            raise Exception('Selected method_name is not defined')