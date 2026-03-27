import ctypes
from ._ffi import _lib


class BPETokenizer:
    def __init__(self, handle: int):
        self._handle = handle

    @classmethod
    def train(cls, corpus: str, max_vocab_size: int = 256) -> "BPETokenizer":
        corpus_bytes = corpus.encode("utf-8")
        handle = _lib.BPE_Train(corpus_bytes, len(corpus_bytes), max_vocab_size)
        if handle < 0:
            raise RuntimeError("BPE training failed")
        return cls(handle)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        handle = _lib.BPE_Load(path.encode("utf-8"))
        if handle < 0:
            raise FileNotFoundError(f"Failed to load tokenizer from {path}")
        return cls(handle)

    def save(self, path: str) -> None:
        rc = _lib.BPE_Save(self._handle, path.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"Failed to save tokenizer to {path}")

    def encode(self, text: str) -> list[int]:
        out_len = ctypes.c_int(0)
        ptr = _lib.BPE_Encode(
            self._handle, text.encode("utf-8"), ctypes.byref(out_len)
        )
        if out_len.value == 0:
            return []
        try:
            return [ptr[i] for i in range(out_len.value)]
        finally:
            _lib.BPE_FreeTokens(ptr)

    def decode(self, ids: list[int]) -> str:
        c_arr = (ctypes.c_int * len(ids))(*ids)
        raw_ptr = _lib.BPE_Decode(self._handle, c_arr, len(ids))
        result = ctypes.cast(raw_ptr, ctypes.c_char_p).value.decode("utf-8")
        _lib.BPE_FreeString(raw_ptr)
        return result

    def close(self):
        if self._handle >= 0:
            _lib.BPE_Free(self._handle)
            self._handle = -1

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
