import ctypes
import os
from pathlib import Path

class CudaConfig:

    def __init__(self, block_size: int | None = None):
        
        lib_path = Path(__file__).parent / 'libcuda_config.so'
        self._lib = ctypes.CDLL(str(lib_path))
        
        self._lib.get_optimal_block_size.restype = ctypes.c_int
        self._lib.get_current_block_size.restype = ctypes.c_int
        self._lib.set_block_size.argtypes = [ctypes.c_int]
        
        if block_size is not None:
            set_block_size(block_size)

    @staticmethod
    def get_block_size() -> int:
        return CudaConfig()._lib.get_current_block_size()
    
    @staticmethod
    def set_block_size(size: int):
        CudaConfig()._lib.set_block_size(size)
    
    @staticmethod
    def get_optimal_block_size() -> int:
        return CudaConfig()._lib.get_optimal_block_size()

