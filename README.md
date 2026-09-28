# byteqc-wheels

**Unofficial** installable builds of [ByteQC](https://github.com/bytedance/byteqc),
ByteDance's GPU quantum chemistry package, with its CUDA kernels precompiled.
This project is not affiliated with or endorsed by ByteDance. ByteQC itself
is © ByteDance Ltd. and its contributors, licensed under Apache-2.0.

## Install

Pick the wheel from the [latest release](https://github.com/XKainz/byteqc-wheels/releases/latest)
and install it into a virtual environment:

```bash
uv venv -p 3.12 byteqc-env
uv pip install -p byteqc-env/bin/python "<wheel URL from the release page>"
```

All CUDA libraries (CuPy, cuBLAS, cuTENSOR) come from PyPI. You need no CUDA
toolkit and no `LD_LIBRARY_PATH`.

## Requirements

- Linux x86-64 with glibc ≥ 2.28 (RHEL/Rocky 8+, Ubuntu 20.04+, Debian 11+)
- NVIDIA driver supporting CUDA 12 (≥ 525)
- Python ≥ 3.11
- An **A100 or H100** GPU. Kernels are also compiled for V100 (sm_70), but
  cuTENSOR contractions fail there with `CUTENSOR_STATUS_NOT_SUPPORTED`, which
  also happens with a from-source build.

## Versions

ByteQC has no tagged releases, so builds are identified by the upstream commit:
`0.0.<commit date>+g<commit>.cu129`. For example, `0.0.20260707+g02af254.cu129`
is ByteQC commit `02af254` from 2026-07-07, built with CUDA 12.9. At runtime:

```python
import byteqc._build_info as b; print(b.__version__, b.upstream_commit)
```

## Modifications to upstream ByteQC

The wheel contains upstream ByteQC unchanged except for:

1. `byteqc/_preload.py` (new): loads the pip-installed cuBLAS/cuTENSOR libraries
   so they are found without `LD_LIBRARY_PATH`.
2. `byteqc/__init__.py`: one added line importing `_preload`, marked with a comment.
3. `byteqc/_build_info.py` (new): version and build metadata.
4. The compiled kernels (`byteqc/*/lib/*.so`) have their library search path
   (RUNPATH) set relative to the install location with `patchelf`.
5. C/CUDA sources and build scripts are omitted. Get them from upstream.

## How it's built

`.github/workflows/build.yml` checks out ByteQC at a pinned commit, compiles
the kernels for sm_70/80/90 in a `manylinux_2_28` container with CUDA 12.9,
packages them with `scripts/package.sh`, verifies the wheel, and attaches it to
a GitHub release when a `v*` tag is pushed. Compiled kernels are cached, so
packaging-only changes don't recompile.

`PACKAGING_GUIDE.md` explains the concepts involved (wheels, dynamic linking,
manylinux, CUDA) from the ground up.

## License

The build scripts in this repository are Apache-2.0. The wheels contain ByteQC
under its original license (`byteqc/LICENSE`, `byteqc/COPYING.APACHE`),
including the notices for code it adapted from PySCF, GPU4PySCF, libcint,
Vayesta (Apache-2.0) and CuPy (MIT).
