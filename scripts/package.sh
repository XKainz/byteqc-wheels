#!/usr/bin/env bash
# Turn a byteqc checkout with compiled .so files into a relocatable wheel.
set -euxo pipefail
cp overlay/_preload.py byteqc/_preload.py
# make the preload run before anything imports cupy
sed -i 's/^from byteqc import lib$/from byteqc import _preload  # noqa: F401\nfrom byteqc import lib/' byteqc/__init__.py
grep -q "_preload" byteqc/__init__.py
cp byteqc/README.md README.md

# RUNPATH relative to the installed location: site-packages/byteqc/<sub>/lib/*.so
RP='$ORIGIN:$ORIGIN/../../../cutensor/lib:$ORIGIN/../../../nvidia/cublas/lib'
for f in byteqc/cuobc/lib/*.so byteqc/cupbc/lib/*.so; do
  patchelf --set-rpath "$RP" "$f"
  # fail if any binary needs a newer glibc than manylinux_2_28 allows
  v=$(objdump -T "$f" | grep -oE 'GLIBC_[0-9.]+' | sed 's/GLIBC_//' | sort -V | tail -1)
  printf '%s\n2.28\n' "$v" | sort -V -C || { echo "$f needs GLIBC_$v"; exit 1; }
done

uv build --wheel -o dist
uvx wheel tags --remove --python-tag py3 --abi-tag none --platform-tag manylinux_2_28_x86_64 dist/*.whl
ls -la dist
