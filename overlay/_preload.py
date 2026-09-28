# Preload CUDA libraries shipped as pip wheels (cutensor-cu12, nvidia-cublas-cu12)
# so cupy's cutensor bindings and byteqc's compiled kernels can resolve them
# without LD_LIBRARY_PATH.
import ctypes
import os
import site
import sys

_LIBS = [
    ('nvidia/cublas/lib', ['libcublasLt.so.12', 'libcublas.so.12']),
    ('cutensor/lib', ['libcutensor.so.2', 'libcutensorMg.so.2']),
]


def _preload():
    roots = list(dict.fromkeys(site.getsitepackages() + [site.getusersitepackages()] + sys.path))
    for sub, names in _LIBS:
        for root in roots:
            d = os.path.join(root, sub)
            if os.path.isdir(d):
                for n in names:
                    p = os.path.join(d, n)
                    if os.path.exists(p):
                        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                break


_preload()
