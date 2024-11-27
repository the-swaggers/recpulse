import ctypes
import dtypes
import numpy as np


def shape_validator(shape) -> tuple[int]:
    try:
        return tuple(i for i in shape if isinstance(i, int) and i > 0)
    except TypeError:
        raise Exception("Shape must be an iterable of positive integers.")

class Tensor:

    def __init__(self, 
        vals,
        shape = None,
        dtype = float,
        device = "cpu"
    ) -> None:

        self.__shape = shape_validator(shape)
        self.__dtype = dtype if not isinstance(dtype, float) else dtypes.float64
        self.__device = "cpu"

        self.__lib = ctypes.CDLL('./libtensor.so')

        self.__lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.__lib.create_tensor.restype = ctypes.c_void_p
        
        self.__lib.fill_tensor.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.__lib.free_tensor.argtypes = [ctypes.c_void_p]
        
        c_shape = (ctypes.c_int * len(self.__shape))(*self.__shape)
        
        self.__tensor_ptr = self.__lib.create_tensor(c_shape, len(self.__shape))
        
        if isinstance(vals, (list, tuple, np.ndarray)):
            self.fill(float(vals[0]) if len(vals) > 0 else 0.0)
    
    def fill(self, value: float) -> None:
        self.__lib.fill_tensor(self.__tensor_ptr, ctypes.c_float(value))
    
    def __del__(self):
        if hasattr(self, '_Tensor__tensor_ptr'):
            self.__lib.free_tensor(self.__tensor_ptr)

