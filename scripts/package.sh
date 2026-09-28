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

# The wheel must contain every file of the byteqc checkout that we meant to ship
# (all .py files, the preload and the kernels), not just some of them.
W=$(ls dist/*.whl)
python3 - "$W" <<'PY'
import sys, zipfile, pathlib
names = set(zipfile.ZipFile(sys.argv[1]).namelist())
want = {str(p) for p in pathlib.Path("byteqc").rglob("*")
        if p.is_file() and (p.suffix in {".py", ".so"}) and "__pycache__" not in p.parts
        and p.name != "setup.py"}  # build scripts, excluded on purpose
missing = sorted(want - names)
if missing:
    sys.exit(f"{len(missing)} files missing from wheel, e.g. {missing[:5]}")
print(f"wheel contains all {len(want)} .py/.so files")
PY
ls -la dist
