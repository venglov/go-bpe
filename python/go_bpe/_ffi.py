import ctypes
import os
import sys

_dir = os.path.dirname(__file__)
_ext = "dylib" if sys.platform == "darwin" else "so"
_lib = ctypes.CDLL(os.path.join(_dir, f"libbpe.{_ext}"))

_lib.BPE_Train.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_lib.BPE_Train.restype = ctypes.c_int

_lib.BPE_Load.argtypes = [ctypes.c_char_p]
_lib.BPE_Load.restype = ctypes.c_int

_lib.BPE_Save.argtypes = [ctypes.c_int, ctypes.c_char_p]
_lib.BPE_Save.restype = ctypes.c_int

_lib.BPE_Encode.argtypes = [
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int),
]
_lib.BPE_Encode.restype = ctypes.POINTER(ctypes.c_int)

_lib.BPE_Decode.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.BPE_Decode.restype = ctypes.c_void_p

_lib.BPE_Free.argtypes = [ctypes.c_int]
_lib.BPE_Free.restype = None

_lib.BPE_FreeTokens.argtypes = [ctypes.POINTER(ctypes.c_int)]
_lib.BPE_FreeTokens.restype = None

_lib.BPE_FreeString.argtypes = [ctypes.c_void_p]
_lib.BPE_FreeString.restype = None
