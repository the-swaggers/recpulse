from __future__ import annotations
import ctypes
import dtypes
import numpy as np


def shape_validator(shape) -> tuple[int]:
    try:
        return tuple(i for i in shape if isinstance(i, int) and i > 0)
    except TypeError:
        raise Exception("Shape must be an iterable of positive integers.")

def flatten(arr: list) -> list:
    flattened = []

    if not isinstance(arr[0], (list, tuple)):
        return arr

    for item in arr:
        flattened.extend(flatten(item))
    
    return flattened

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
        
        self.__lib.fill_tensor_scalar.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.__lib.fill_tensor_vals.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.__lib.free_tensor.argtypes = [ctypes.c_void_p]
        
        self.__lib.element_from_tensor.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.__lib.element_from_tensor.restype = ctypes.c_float
        
        self.__lib.vals_from_tensor.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        
        self.__lib.free_array.argtypes = [ctypes.POINTER(ctypes.c_float)] 

        c_shape = (ctypes.c_int * len(self.__shape))(*self.__shape)
        
        self.__tensor_ptr = self.__lib.create_tensor(c_shape, len(self.__shape))
        
        vals = flatten(vals)
        new_vals = (ctypes.c_float * len(vals))(*vals)

        arr1 = self.to_numpy()

        self.fill_vals(new_vals)
        
        arr2 = self.to_numpy()

        print(arr1, arr2)

    def fill_vals(self, new_vals) -> None:
        self.__lib.fill_tensor_vals(self.__tensor_ptr, new_vals)
    
    def __del__(self):
        if hasattr(self, '_Tensor__tensor_ptr'):
            self.__lib.free_tensor(self.__tensor_ptr)
    
    def __getitem__(self, idx: int | slice | tuple[int | slice]) -> Tensor | ctypes.c_float:
        # for now slices unsupported
        
        if isinstance(idx, (int, slice)):
            idx = (idx, )
        
        print(idx)
        c_idx = (ctypes.c_int * len(self.__shape))(*idx)
        print(c_idx)
        return self.__lib.element_from_tensor(self.__tensor_ptr, c_idx)
    
    @property
    def size(self):
        size = 1
        for i in self.__shape:
            size *= i
        return size

    def to_numpy(self):

        tmp = (self.size * ctypes.c_float)()

        self.__lib.vals_from_tensor(self.__tensor_ptr, tmp)

        numpy_array = np.ctypeslib.as_array(tmp, shape=(self.size,))
        numpy_copy = numpy_array.copy()

        return numpy_copy

    def __str__(self):
        return str(self.to_numpy())

a = Tensor([[0, 1], [2, 3]], shape=(2,2))

print("adas")

arr = a.to_numpy()

print(arr)

print(a[0, 0])

