# byteqc-wheels

Builds an installable wheel of [ByteQC](https://github.com/bytedance/byteqc)
(pinned commit in `.github/workflows/build.yml`) with its CUDA kernels
precompiled for sm_70/80/90 (V100/A100/H100). CUDA libraries come from pip,
so users need only an NVIDIA driver (CUDA 12.9 capable).

    uv venv && uv pip install byteqc-*.whl

Build: Actions -> build-wheel -> Run workflow (or push a `v*` tag to also
attach the wheel to a GitHub release).
