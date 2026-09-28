#!/usr/bin/env bash
# Compile one byteqc subpackage's CUDA kernels (cuobc or cupbc).
# Runs inside the manylinux_2_28 container; see .github/workflows/build.yml.
set -euxo pipefail
SUB=$1
JOBS=${JOBS:-2}
ARCHS=${CUDA_ARCHS:-70;80;90}

# CUDA 12.9 compiler + cuBLAS headers from NVIDIA's RHEL8 repo (manylinux_2_28 is AlmaLinux 8)
curl -fsSL -o /etc/yum.repos.d/cuda-rhel8.repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y cuda-nvcc-12-9 cuda-cudart-devel-12-9 libcublas-devel-12-9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc

# cuTENSOR: the same pip package the wheel depends on at runtime (it ships headers too).
# byteqc's CMakeLists only searches $CUTENSOR_ROOT/{include,lib/12}, so pass the
# header and library paths directly as CMake cache variables.
PY=/opt/python/cp312-cp312/bin/python
$PY -m pip install -q "cutensor-cu12==2.6.0"
CT=$($PY -c "import cutensor, os; print(os.path.dirname(cutensor.__file__))")
CUTENSOR_ARGS=(
  -DCUTENSOR_LIB=$CT/lib/libcutensor.so.2
  -DCUTENSOR_HEADER=$CT/include/cutensor.h
)

LIB=byteqc/$SUB/lib
cmake -S $LIB -B $LIB/build -DCUDA_ARCHITECTURES="$ARCHS" -DCMAKE_BUILD_TYPE=Release "${CUTENSOR_ARGS[@]}"
# -j is kept low: fill_int2e.cu (cupbc) needs several GB of RAM per nvcc process
cmake --build $LIB/build -j "$JOBS"
ls -la $LIB/*.so
