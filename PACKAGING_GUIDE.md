# Python Packaging and Native Code, From the Ground Up

A small textbook, written while building `byteqc-wheels`. It starts at "what
does `import` do" and ends at "ship a CUDA library to anyone with `uv pip
install`". Every concept is tied back to a concrete thing we hit in this
project, including the mistakes.

---

## Contents

1. What `import` actually does
2. Packages vs. distributions vs. environments
3. Installers, resolvers and dependency metadata
4. Wheels: what they are, byte by byte
5. Building wheels: `pyproject.toml`, frontends and backends
6. Compiled code: object files, shared libraries, and how Python loads them
7. Dynamic linking in depth (the part everything else depends on)
8. glibc, symbol versions and manylinux
9. CUDA specifics: driver, runtime, toolkit, architectures
10. How the big projects do it (PyTorch, JAX, CuPy)
11. Building in CI: containers and GitHub Actions
12. Distributing: files, releases, indexes, PyPI
13. The byteqc-wheels pipeline, file by file
14. Debugging cookbook
15. Recipe for next time
16. Glossary

---

## 1. What `import` actually does

When Python runs `import byteqc`, it walks the list `sys.path` in order and
looks in each directory for either

- a folder `byteqc/` (a *package*), or
- a file `byteqc.py` (a *module*), or
- a compiled extension `byteqc.cpython-312-x86_64-linux-gnu.so`.

The first hit wins. That is the entire mechanism. Everything in packaging is
ultimately about getting the right files into a directory that is on
`sys.path`.

```
>>> import sys; print(*sys.path, sep="\n")
                                         <- current directory / script dir
/usr/lib/python313.zip
/usr/lib/python3.13
/usr/lib/python3.13/lib-dynload
/home/me/venv/lib/python3.13/site-packages   <- where installers put things
```

**`site-packages`** is the conventional directory installers write into.

**`.pth` files.** At startup, Python's `site` module reads every `*.pth` file
in `site-packages`. Each line that is a path is appended to `sys.path`; each
line starting with `import` is executed. Your original cluster setup used
exactly this: `testvenv/.../site-packages/_byteqc_path.pth` contains the single
line `/ceph/ssd/students/kaix/opt`, which puts the *parent* of the `byteqc/`
folder on `sys.path`, so `import byteqc` finds it. That is a hand-made
"editable install".

**Packages with and without `__init__.py`.** A folder with `__init__.py` is a
regular package; the file runs on first import, and `pkg.__file__` points at
it. A folder *without* `__init__.py` is a **namespace package** (PEP 420): it
still imports, but `pkg.__file__` is `None` and its location is only in
`pkg.__path__`. The NVIDIA pip wheels (`cutensor`, `nvidia.cublas`, ...) are
namespace packages. Our first CI run died on exactly this:

```python
os.path.dirname(cutensor.__file__)     # TypeError: ... not NoneType
list(cutensor.__path__)[0]             # correct
```

---

## 2. Packages vs. distributions vs. environments

Three words that are constantly confused:

| Term | What it is | Example |
|---|---|---|
| **import package** | what you write after `import` | `cupy`, `cutensor`, `byteqc` |
| **distribution** (a.k.a. "project", "the thing you pip install") | a named, versioned bundle that installs one or more import packages | `cupy-cuda12x`, `cutensor-cu12`, `nvidia-cublas-cu12` |
| **environment** | one Python interpreter + one `site-packages` | a venv |

The names do not have to match: installing the distribution `cupy-cuda12x`
gives you the import package `cupy`; `nvidia-cublas-cu12` gives you
`nvidia/cublas/`. This is how CuPy ships one import name but many builds
(`cupy-cuda11x`, `cupy-cuda12x`, `cupy-rocm-*`).

**Virtual environments.** A venv is just a directory with a `bin/python`
(symlink to a real interpreter), its own `site-packages`, and a
`pyvenv.cfg`. Activating it only puts its `bin/` first on `PATH`. Two venvs
never see each other's packages, which is why we tested the wheel in a *fresh*
venv: it proves nothing is leaking in from your existing setup.

---

## 3. Installers, resolvers and dependency metadata

`pip` and `uv` are **installers**. Given a requirement like
`byteqc` or `cupy-cuda12x>=14,<15` they:

1. Find candidate files on an **index** (PyPI by default) or a local path.
2. Read each candidate's **metadata**: its version and its own dependencies.
3. **Resolve**: pick one version of every distribution so that all constraints
   hold at once. This is a constraint-satisfaction problem; uv is fast at it.
4. Download the chosen **wheels** and unpack them into `site-packages`.

Dependencies are declared as *requirement specifiers* (PEP 508):

```
cupy-cuda12x[ctk]>=14,<15     name, optional "extra" in [], version range
pyscf>=2.14,<2.15             allow bugfixes, not the next minor release
numpy                         anything
gpu4pyscf-cuda12x ; python_version >= "3.11"   with an environment marker
```

An **extra** (`[ctk]`) is an optional group of additional dependencies the
package author defined. `cupy-cuda12x[ctk]` pulls in NVIDIA's pip-packaged
CUDA runtime libraries so CuPy doesn't need a system CUDA install.

**Versions** follow PEP 440: `1.2.3`, `1.2.3rc1`, `1.2.3.post1`, and *local
versions* `1.2.3+cu129`. The `+local` part is for builds that are "the same
release, built differently". PyPI **rejects** local versions; they are meant
for your own indexes and files. We use `0.0.20260707+g02af254.cu129` (upstream commit date + hash + CUDA; ByteQC has no releases), which is fine for
GitHub releases and direct installs.

---

## 4. Wheels: what they are, byte by byte

A **wheel** (`.whl`, PEP 427) is a **zip file** whose contents are laid out
exactly as they should appear in `site-packages`, plus a metadata folder.
Installing a wheel is essentially "unzip into site-packages". No code runs,
nothing compiles. That is the whole point.

```
byteqc-0.0.20260707+g02af254.cu129-py3-none-manylinux_2_28_x86_64.whl   (a zip)
├── byteqc/                        <- copied to site-packages/byteqc/
│   ├── __init__.py
│   ├── _preload.py
│   ├── cuobc/lib/libgint.so       <- compiled GPU kernels, just files
│   └── ...
└── byteqc-0.0.20260707+g02af254.cu129.dist-info/  <- metadata
    ├── METADATA                   <- name, version, Requires-Dist: ...
    ├── WHEEL                      <- wheel format version, tags
    ├── RECORD                     <- every file + sha256, for uninstall/verification
    └── licenses/...
```

Try it: `python -m zipfile -l some.whl`.

### The filename *is* the compatibility contract

```
{name}-{version}(-{build})?-{python tag}-{abi tag}-{platform tag}.whl
byteqc-0.0.20260707+g02af254.cu129-py3-none-manylinux_2_28_x86_64.whl
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^ ^^^^ ^^^^^^^^^^^^^^^^^^^^^
       version                     |   |    platform: Linux, x86-64, glibc >= 2.28
                                   |   ABI: "none" = doesn't use the CPython C ABI
                                   python: any Python 3
```

The installer only considers wheels whose tags match the running system.

| Kind | Example tags | Meaning |
|---|---|---|
| pure Python | `py3-none-any` | runs anywhere |
| CPython extension | `cp312-cp312-manylinux_2_28_x86_64` | compiled against CPython 3.12's C API; one wheel *per Python version* |
| stable-ABI extension | `cp311-abi3-manylinux_2_28_x86_64` | compiled against the limited API; works on 3.11+ |
| native code, no Python API | `py3-none-manylinux_2_28_x86_64` | **us**: `.so` files loaded via `ctypes`, so any Python 3, but only Linux x86-64 |

byteqc's `.so` files don't `#include <Python.h>`; Python calls them through
`ctypes`. So a single wheel works for Python 3.11, 3.12, 3.13, ... That's why
we retag with `py3-none-...` instead of building per Python version.

### Wheels vs. source distributions

A **source distribution** (sdist, `.tar.gz`) is the source code plus a recipe.
Installing one means *building it on the user's machine*. For byteqc that
would mean the user needs `nvcc`, cuTENSOR headers, CMake, an hour of compile
time and enough RAM. Wheels exist to move that work to the publisher, once.

---

## 5. Building wheels: `pyproject.toml`, frontends and backends

Building a wheel is split between two tools (PEP 517/518):

- **Build frontend**: the thing you run. `uv build`, `python -m build`,
  `pip wheel`. It creates an isolated env, installs the backend, and calls it.
- **Build backend**: the library that knows how to turn *this* project into a
  wheel. Declared in `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]            # installed into the isolated build env
build-backend = "hatchling.build"   # the module the frontend calls
```

Common backends:

| Backend | Use for |
|---|---|
| `hatchling`, `flit-core`, `setuptools` | pure Python, or packaging already-compiled files |
| `setuptools` + `Extension` | classic C extensions |
| `scikit-build-core` | projects built with **CMake** (compiles during the build) |
| `meson-python` | projects built with Meson (NumPy, SciPy) |
| `maturin` | Rust |

Project metadata lives in the standard `[project]` table (PEP 621):

```toml
[project]
name = "byteqc"
dynamic = ["version"]   # read from byteqc/_build_info.py (see scripts/package.sh)
requires-python = ">=3.11"
dependencies = ["cupy-cuda12x[ctk]>=14,<15", "cutensor-cu12>=2.6,<3", ...]
```

### Our choice: compile first, package second

We compile with CMake in one CI job and let `hatchling` merely *collect*
files. That's simpler than wiring byteqc's non-standard CMake into
scikit-build-core, and it lets the slow compile happen in parallel jobs.
The trade-off: `hatchling` doesn't know the files are platform-specific and
names the wheel `py3-none-any`, so we **retag** it afterwards with
`wheel tags --platform-tag manylinux_2_28_x86_64`. (A `py3-none-any` wheel
containing Linux binaries would happily install on macOS and then crash.)

### Gotcha we hit: `.gitignore`

Hatchling respects `.gitignore` when choosing files. byteqc's own
`.gitignore` contains `*.so`, so the compiled kernels would silently be left
out. The fix is `artifacts = ["byteqc/**/*.so"]`, which means "include these
even if ignored". Always look inside a freshly built wheel
(`python -m zipfile -l`) before trusting it.

---

## 6. Compiled code: object files, shared libraries, and loading

### From source to library

```
fill_int2e.cu --nvcc--> fill_int2e.o ─┐
util.cu       --nvcc--> util.o       ─┼--linker--> libgpbc.so
get_Rcuts.cu  --nvcc--> get_Rcuts.o  ─┘                ^ shared library
```

- **Compiling** turns each source file into an **object file** (`.o`):
  machine code with holes where it calls functions defined elsewhere.
- **Linking** combines objects into a final artifact and decides how to fill
  the holes:
  - **static linking**: copy the needed code *into* the output (`.a`
    archives). Bigger file, no runtime dependency. CMake links the CUDA
    runtime (`cudart`) statically by default, which is why our `.so` files
    don't list `libcudart.so`.
  - **dynamic linking**: leave a note "at runtime, load `libcublas.so.12`
    and find `cublasDgemm` in it". Smaller, shareable, but now there's a
    runtime dependency that must be findable.

A **shared library** (`.so` on Linux, `.dll` on Windows, `.dylib` on macOS)
is the result of dynamic-linkable code: loaded into a process at runtime.

### How Python loads one

Two ways:

1. **Extension module**: a `.so` that exports `PyInit_<name>`. `import x`
   loads it directly. Tied to the CPython ABI, hence the `cp312` tags.
2. **`ctypes`**: plain C library, loaded explicitly:
   ```python
   lib = ctypes.CDLL("/path/libgint.so")          # or numpy.ctypeslib.load_library
   lib.some_function(ctypes.c_int(3), ptr)
   ```
   byteqc does this (`byteqc/cuobc/lib/__init__.py:load_library`). Under the
   hood both use the OS function `dlopen()`, so both are subject to all the
   linking rules in the next section.

---

## 7. Dynamic linking in depth

This is where almost every "works on my machine" problem with native wheels
lives. Linux-specific (ELF, glibc's `ld.so`), but macOS and Windows have
equivalents.

### 7.1 What's written inside a `.so`

Every ELF shared library has a *dynamic section* you can print:

```
$ readelf -d libgpbc.so
 (NEEDED)   Shared library: [libcublas.so.12]
 (NEEDED)   Shared library: [libcutensor.so.2]
 (NEEDED)   Shared library: [libstdc++.so.6]
 (NEEDED)   Shared library: [libc.so.6]
 (RUNPATH)  Library runpath: [$ORIGIN:$ORIGIN/../../../cutensor/lib:...]
```

- **NEEDED**: the libraries this one depends on, *by name*, not by path.
- **SONAME** (on the library being depended upon): its official name.
  `libcutensor.so.2.6.0` declares `SONAME = libcutensor.so.2`.
- **RUNPATH / RPATH**: extra directories to search for this library's
  NEEDED entries.

### 7.2 The three names of a library

```
libcutensor.so        -> symlink, used only at BUILD time by the linker (-lcutensor)
libcutensor.so.2      -> the SONAME; what gets recorded in NEEDED; used at RUN time
libcutensor.so.2.6.0  -> the actual file
```

When you link with `-lcutensor`, the linker looks for `libcutensor.so`, opens
it, reads its SONAME (`libcutensor.so.2`), and writes **that** into your
library's NEEDED. The major number in the soname is the ABI promise: any
`libcutensor.so.2.x` can satisfy it.

Consequences we hit:

- **"devel" packages** (`libcublas-devel-12-9`, `*-dev` on Debian) exist
  mainly to provide headers and the bare `libfoo.so` symlink needed at build
  time. Runtime packages ship only `libfoo.so.N*`.
- **The pip `cutensor-cu12` wheel has no bare `libcutensor.so`**, only
  `libcutensor.so.2`. That's why byteqc's CMake `find_library(... cutensor)`
  couldn't find it, and why passing the full path
  `-DCUTENSOR_LIB=.../libcutensor.so.2` works: the linker still reads the
  SONAME and records `libcutensor.so.2`.
- **Your original cluster build had a latent bug**: `libgpbc.so`'s RUNPATH
  pointed at `opt/cutensor_root/lib/12/`, which contained only the *bare*
  `libcutensor.so` symlink. At runtime the loader looks for
  `libcutensor.so.2`, which wasn't there, so `ldd` said `not found`. It only
  worked because something else had loaded cuTENSOR first (see 7.4).

### 7.3 Where the loader looks

When a library is loaded, the dynamic loader (`ld-linux-x86-64.so.2`)
resolves each NEEDED name in this order (glibc):

1. **Already loaded?** If a library with that SONAME is already in the
   process, reuse it. (Section 7.4.)
2. **DT_RPATH** of the requesting library (legacy; ignored if RUNPATH is
   set).
3. **`LD_LIBRARY_PATH`** environment variable.
4. **DT_RUNPATH** of the requesting library. Note: RUNPATH applies only to
   that library's *direct* dependencies, not transitively.
5. **`/etc/ld.so.cache`**: system libraries registered by `ldconfig`.
6. Default dirs: `/lib`, `/usr/lib` (and 64-bit variants).

`$ORIGIN` in an RPATH/RUNPATH means "the directory containing this `.so`".
That's what makes a library **relocatable**: it can find its neighbours
wherever it's installed.

In a wheel installed to `site-packages/`:

```
site-packages/
├── byteqc/cupbc/lib/libgpbc.so        <- $ORIGIN is here
├── cutensor/lib/libcutensor.so.2      <- $ORIGIN/../../../cutensor/lib
└── nvidia/cublas/lib/libcublas.so.12  <- $ORIGIN/../../../nvidia/cublas/lib
```

We set that with **`patchelf --set-rpath`**, a tool that edits the dynamic
section of an existing ELF file. Before, `libgvhf.so` had
`RUNPATH=/ceph/ssd/students/kaix/opt/byteqc/cuobc/lib`: an absolute path into
your home. Copy it to another user and it would silently load *your* copy of
`libgint.so` as long as yours existed.

### 7.4 Preloading, and why `LD_LIBRARY_PATH` was needed

Rule 1 above (reuse by SONAME) is powerful: if you load
`.../cutensor/lib/libcutensor.so.2` by full path *first*, every later library
that NEEDs `libcutensor.so.2` gets that one, regardless of search paths.

This is how PyTorch finds its pip-installed CUDA libraries, and what
`byteqc/_preload.py` does:

```python
ctypes.CDLL(".../site-packages/cutensor/lib/libcutensorMg.so.2", mode=ctypes.RTLD_GLOBAL)
```

(`RTLD_GLOBAL` additionally makes the library's *symbols* available to
libraries loaded later; harmless here and sometimes necessary.)

Why was it needed at all? CuPy's compiled module
`cupy_backends/cuda/libs/cutensor.so` NEEDs `libcutensorMg.so.2`, but CuPy
doesn't know where pip put cuTENSOR. Without help, the loader searched
`LD_LIBRARY_PATH` → system → failed:

```
ImportError: libcutensorMg.so.2: cannot open shared object file
```

Your working setup fixed that with `export LD_LIBRARY_PATH=...`. It works,
but every user has to remember it in every shell and every sbatch script.
Preloading in `byteqc/__init__.py` does the same thing automatically.

### 7.5 Symbols and symbol versions

Beyond library names, the loader resolves individual **symbols**
(functions/variables). Missing ones give `undefined symbol: foo`. Usually a
version mismatch: built against a newer library than the one loaded.

glibc additionally **versions** its symbols: `memcpy@GLIBC_2.14`. A binary
records the version it was linked against, and refuses to load if the
running glibc is older (Section 8).

### 7.6 Toolbox

| Command | Shows |
|---|---|
| `ldd lib.so` | every dependency and where it *would* resolve right now (`not found` = problem) |
| `readelf -d lib.so` | NEEDED, SONAME, RPATH/RUNPATH |
| `objdump -T lib.so \| grep GLIBC_` | glibc symbol versions required |
| `patchelf --print-rpath / --set-rpath` | read/change RUNPATH after building |
| `LD_DEBUG=libs python -c "import x"` | the loader narrating every search it does |
| `nm -D lib.so` | exported/imported symbols |

`LD_DEBUG=libs` is the single most useful debugging tool in this entire
document.

---

## 8. glibc, symbol versions and manylinux

### 8.1 The problem

Every Linux program links dynamically to **glibc** (`libc.so.6`). glibc is
**backward compatible but not forward compatible**:

- built against glibc 2.17 → runs on 2.17, 2.28, 2.39 ✔
- built against glibc 2.34 → on a 2.28 system: `version 'GLIBC_2.34' not found` ✘

You automatically link against whatever glibc the *build machine* has. Your
cluster (Ubuntu 24.04, glibc 2.39) produced `.so` files needing 2.34. The
CI build (AlmaLinux 8, glibc 2.28) produced files needing only **2.17**.

So the rule is: **build on the oldest system you want to support.**

### 8.2 manylinux

A wheel tagged `manylinux_X_Y_arch` (PEP 600) promises:

1. it needs glibc ≥ X.Y, and
2. it links only against a short allowlist of "everyone has these" system
   libraries (`libc`, `libm`, `libpthread`, `libdl`, `librt`, `libgcc_s`,
   `libstdc++` up to a version, a few X11/GL libs). **Anything else must be
   bundled in the wheel or come from another wheel.**

Common targets:

| Tag | Build image | Runs on |
|---|---|---|
| `manylinux2014` = `manylinux_2_17` | CentOS 7 | practically everything; image is EOL |
| `manylinux_2_28` | AlmaLinux 8 | RHEL/Rocky 8+, Ubuntu 20.04+, Debian 11+ |
| `manylinux_2_34` | AlmaLinux 9 | RHEL 9+, Ubuntu 22.04+ |

The **PyPA manylinux images** (`quay.io/pypa/manylinux_2_28_x86_64`) are
those old systems plus: every Python version under `/opt/python/`, and a
modern GCC (`gcc-toolset-14`) configured so that it only emits
old-glibc/old-libstdc++-compatible code. (It links new C++ features
statically via `libstdc++_nonshared.a`.) You get a modern compiler with old
compatibility.

### 8.3 auditwheel

`auditwheel show x.whl` checks a wheel's `.so` files against the policies
and tells you the best tag it qualifies for. `auditwheel repair` fixes a
wheel by **copying** every non-allowlisted dependency into it (`x.libs/`),
renaming them to avoid clashes, and patching RUNPATHs. The standard tool.

We **don't** use `repair`, deliberately: it would copy ~1 GB of cuBLAS and
cuTENSOR into the wheel, and CuPy would load its own copies anyway (two
cuBLASes in one process is asking for trouble). Instead we depend on
NVIDIA's pip wheels and point RUNPATH at them. `package.sh` does the glibc
check itself; the `check` CI job verifies `ldd` finds everything.

---

## 9. CUDA specifics

### 9.1 The four layers

```
 your code (byteqc kernels, CuPy)
      │ calls
 CUDA libraries: cuBLAS, cuTENSOR, cuFFT, ...    <- pip wheels (nvidia-*-cu12, cutensor-cu12)
      │ call
 CUDA runtime: libcudart                          <- statically linked into our .so
      │ calls
 CUDA driver: libcuda.so.1 + kernel module        <- installed by the admin, NEVER shipped
      │
 GPU
```

- The **driver** comes with the machine. You can't pip-install it and must
  never bundle `libcuda.so`. Check with `nvidia-smi` (top right: highest CUDA
  version the driver supports).
- The **toolkit** (`nvcc`, headers, dev libraries) is only needed to
  **build**. Users don't need it.
- The **runtime libraries** are needed to **run**; NVIDIA publishes them as
  pip wheels (`nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`,
  `cutensor-cu12`, ...), installed into `site-packages/nvidia/...`.

**Compatibility.** Newer drivers run older CUDA builds. Within a major
version, *minor version compatibility* lets a CUDA 12.9 build run on any
CUDA 12 driver ≥ 525, as long as it doesn't need to JIT-compile PTX newer
than the driver understands (see 9.2). The cluster's driver 570 reports
CUDA 12.8, and our 12.9 build still runs because it ships native code for
every GPU there.

### 9.2 GPU architectures, SASS and PTX

GPUs have a **compute capability**: V100 = 7.0 (`sm_70`), A100 = 8.0,
H100 = 9.0, L40/RTX 4090 = 8.9, Blackwell = 10.0/12.0.

`nvcc` can emit two kinds of GPU code:

- **SASS**: native machine code for one architecture. Fast to load; runs on
  that architecture (and same-major newer minors, e.g. sm_80 SASS on sm_86).
- **PTX**: a portable intermediate assembly. The driver JIT-compiles it at
  first use for whatever GPU is present, including future ones, but only if
  the driver is new enough to understand that PTX version.

A **fat binary** bundles several of each into one `.so`. In CMake,
`CUDA_ARCHITECTURES "70;80;90"` means SASS + PTX for each; `"70-real"` would
mean SASS only. We verified the CI artifacts contain code for sm_70/80/90.

If a GPU isn't covered and there's no usable PTX, you get
`no kernel image is available for execution on the device`.

This is why GPU wheels are big: every kernel is compiled once per
architecture. `libgaft.so` alone is ~118 MB.

### 9.3 `nvcc` and host compilers

`nvcc` compiles the GPU parts itself and hands CPU parts to a **host
compiler** (GCC). Each CUDA version supports a range of GCC versions (CUDA
12.9: up to GCC 14). In the manylinux container `gcc-toolset-14` is first on
`PATH`, so that's the one used.

---

## 10. How the big projects do it

| Project | Strategy |
|---|---|
| **PyTorch** | One huge wheel per (Python, CUDA) combo. Since ~2.1 it depends on `nvidia-*-cu12` wheels and **preloads** them by path at import (the same trick as our `_preload.py`). CUDA variants live on separate indexes (`--index-url https://download.pytorch.org/whl/cu124`). |
| **JAX** | `jax` (pure Python) + `jaxlib` (compiled CPU) + plugin wheels `jax-cuda12-plugin`/`jax-cuda12-pjrt`, installed via the extra `jax[cuda12]`, which depends on `nvidia-*` wheels. |
| **CuPy** | Separate distribution per CUDA major (`cupy-cuda12x`). Finds CUDA libs from the system, `CUDA_PATH`, or the `[ctk]` extra's pip wheels. |
| **gpu4pyscf** | `gpu4pyscf-cuda12x` plus `gpu4pyscf-libxc-cuda12x`: kernels in manylinux wheels, CUDA libs from pip. The closest analogue to byteqc. |

Common patterns worth copying:

1. Put the CUDA major in the **distribution name** (`-cuda12x`, `-cu12`).
2. Take CUDA libraries from NVIDIA's pip wheels; never bundle `libcuda`.
3. Make the `.so` files relocatable (`$ORIGIN` RUNPATH) and/or preload deps.
4. Build in manylinux containers in CI.

---

## 11. Building in CI: containers and GitHub Actions

### Why a container

Section 8: build on an old system. You can't install AlmaLinux 8 on the
cluster, but a **container** is a whole OS userland in a box. `docker run
quay.io/pypa/manylinux_2_28_x86_64` gives you AlmaLinux 8 in seconds. Docker
needs root-equivalent rights, which HPC clusters don't give out; GitHub's
hosted runners have it.

### GitHub Actions in five concepts

```yaml
on: workflow_dispatch               # 1. trigger: manual button (also: push, tags, schedule)
jobs:                                # 2. jobs: run in parallel on separate VMs unless `needs:`
  kernels:
    runs-on: ubuntu-latest           #    the VM
    container: quay.io/pypa/...      #    run every step inside this image
    strategy:
      matrix: { sub: [cuobc, cupbc] }  # 3. matrix: one job per value, in parallel
    steps:                           # 4. steps: shell commands or reusable "actions"
      - uses: actions/checkout@v4
      - run: scripts/build_kernels.sh ${{ matrix.sub }}
      - uses: actions/upload-artifact@v4   # 5. artifacts: pass files between jobs / to you
```

Things to know:

- Each job starts on a **fresh VM**; nothing survives except artifacts.
- Free public-repo runners: 4 CPU, 16 GB RAM, ~14 GB free disk, 6 h per job.
  Private repos get 2 CPU / 7 GB and a monthly minutes quota.
- Logs are only downloadable after the whole run finishes, but artifacts are
  available as soon as the job that uploaded them is done (that's how we
  inspected `libgint.so` mid-run).
- `gh run list`, `gh run view`, `gh run download`, `gh workflow run` drive it
  from the terminal.

---

## 12. Distributing

From least to most effort:

| Where | Install command | Notes |
|---|---|---|
| file on shared disk | `uv pip install /path/byteqc-...whl` | zero infrastructure |
| GitHub release asset | `uv pip install https://github.com/.../byteqc-...whl` | our `release` job does this on `v*` tags; 2 GB per file limit |
| your own index | `uv pip install byteqc --index-url https://...` | GitHub Pages "simple" index, or a server |
| **PyPI** | `uv pip install byteqc` | 100 MB/file default (request more), no `+local` versions, name must be free; use *trusted publishing* from Actions |

Always install the wheel into a **fresh** venv on a **different** machine or
account as the final test. That's the only proof it doesn't depend on
something on your machine.

Etiquette: byteqc is ByteDance's project (Apache-2.0, redistribution
allowed, keep the license and attribution). Publishing under *their* name on
PyPI should be coordinated with them. The best outcome is contributing the
workflow upstream.

---

## 13. The byteqc-wheels pipeline, file by file

```
byteqc-wheels/
├── .github/workflows/build.yml   CI orchestration
├── scripts/build_kernels.sh      compile one subpackage inside manylinux
├── scripts/package.sh            make relocatable, build + tag wheel
├── overlay/_preload.py           added to byteqc: preload pip CUDA libs
├── pyproject.toml                wheel metadata & dependencies
└── README.md
```

**Job `kernels` (matrix: cuobc, cupbc), in `manylinux_2_28`:**

1. Check out this repo and upstream byteqc at a pinned commit
   (reproducibility: the same commit always gives the same wheel).
2. `dnf install cuda-nvcc-12-9 cuda-cudart-devel-12-9 libcublas-devel-12-9`
   from NVIDIA's RHEL 8 repo. Build-time only.
3. `pip install cutensor-cu12==2.6.0`; locate it via `__path__`; pass
   `-DCUTENSOR_LIB`/`-DCUTENSOR_HEADER` to CMake. These are CMake *cache
   variables*: `find_library()` skips searching when they're already set,
   which is the standard way to point any CMake project at a dependency
   without editing it.
4. `cmake -DCUDA_ARCHITECTURES="70;80;90"`, compile with low `-j`
   (`fill_int2e.cu` needs several GB of RAM per `nvcc` process; your cluster
   build was OOM-killed, exit code 137 = killed by signal 9).
5. Upload the `.so` files as an artifact.

**Job `wheel`:**

1. Fresh byteqc checkout + both artifacts dropped into place.
2. Add `_preload.py` and import it first in `byteqc/__init__.py`.
3. `patchelf --set-rpath '$ORIGIN:$ORIGIN/../../../cutensor/lib:$ORIGIN/../../../nvidia/cublas/lib'`.
4. Refuse any `.so` needing glibc > 2.28.
5. `uv build --wheel` (hatchling), then retag to `py3-none-manylinux_2_28_x86_64`.

**Job `check`:** fresh venv, install the wheel *with its dependencies*, fail
if `ldd` reports anything `not found` (except `libcuda`, which only exists on
GPU machines). Try an import (allowed to fail: no GPU).

**Job `release`** (only on `v*` tags): attach the wheel to a GitHub release.

**Final test (manual):** install on the cluster in a fresh venv *without*
`LD_LIBRARY_PATH` and run the GPU smoke test on an A100.

---

## 14. Debugging cookbook

| Symptom | Meaning | Look with / fix |
|---|---|---|
| `libX.so.N: cannot open shared object file` | loader couldn't find a NEEDED lib | `ldd`, `LD_DEBUG=libs`; fix RUNPATH, preload, or install the dependency wheel |
| `version 'GLIBC_2.34' not found` | built on a newer system than you're running | rebuild in an older manylinux image |
| `version 'GLIBCXX_3.4.30' not found` | same, for libstdc++ | manylinux toolchain, or a newer libstdc++ first on the path |
| `undefined symbol: foo` | found a library, but the wrong version of it | `nm -D`; check which copy loaded with `LD_DEBUG=libs` |
| `no kernel image is available for execution on the device` | no SASS/PTX for this GPU | add its arch to `CUDA_ARCHITECTURES` |
| `CUDA driver version is insufficient` | toolkit/PTX newer than driver | ship SASS for the GPUs, or build with older CUDA |
| wheel installs but files are missing | backend excluded them (`.gitignore`!) | `python -m zipfile -l x.whl` |
| wheel installs on the wrong platform | wrong tags (`py3-none-any` with binaries) | retag / use a platform-aware backend |
| build killed, exit 137 | out of memory (SIGKILL) | lower `-j` |
| `__file__` is `None` | namespace package | use `__path__` |
| works for you, not for colleague | absolute paths into your home, or your env vars | `readelf -d` (RUNPATH), test in a fresh venv as another user |

---

## 15. Recipe for next time

To ship a Python package with compiled (CUDA) code:

1. **Inventory the binaries.** For each `.so`: `readelf -d` (NEEDED,
   RUNPATH), `objdump -T | grep GLIBC_` (glibc floor). Know what links to what.
2. **Classify every dependency**: allowlisted system lib (fine) / available
   as a pip wheel (depend on it) / neither (bundle it with `auditwheel
   repair`) / the GPU driver (never ship).
3. **Pick the platform tag.** Usually `manylinux_2_28`. Build in the matching
   PyPA image, in CI.
4. **Pick the Python tag.** Extension modules → per-Python (`cp3xx`) or
   `abi3`. `ctypes`-only → `py3-none`.
5. **Pin everything**: upstream commit, toolkit version, library versions.
6. **Make it relocatable**: `$ORIGIN`-relative RUNPATH; preload pip-provided
   libraries that third-party code (like CuPy) needs.
7. **Declare dependencies** in `pyproject.toml`, with sensible ranges.
8. **Build, then look inside the wheel.** File list, tags, `ldd` in a clean
   venv.
9. **Test on a real target** in a fresh environment, as a different user if
   you can, with no `LD_LIBRARY_PATH`.
10. **Distribute** via a file, GitHub release, or index. Coordinate with
    upstream before claiming a PyPI name.

---

## 16. Glossary

- **ABI**: binary interface: calling conventions, struct layouts, symbol
  names. Must match between a library and whatever calls it.
- **Artifact (CI)**: files saved from a CI job.
- **auditwheel**: checks/repairs Linux wheels against manylinux policies.
- **Build backend / frontend**: see §5.
- **Compute capability / sm_XX**: GPU architecture version.
- **Distribution**: an installable, versioned project (what `pip install`
  names).
- **dist-info**: metadata folder of an installed distribution.
- **ELF**: Linux executable/library file format.
- **Extra**: optional dependency group, `pkg[extra]`.
- **Fat binary**: GPU code for several architectures in one file.
- **glibc**: the GNU C library; its version sets the manylinux floor.
- **Index**: server listing distributions (PyPI, or your own).
- **manylinux**: portable Linux wheel standard (PEP 513/599/600).
- **Namespace package**: package without `__init__.py`; no `__file__`.
- **NEEDED**: ELF entry naming a runtime dependency.
- **`$ORIGIN`**: RUNPATH token for "this library's directory".
- **patchelf**: edits RUNPATH/NEEDED of existing ELF files.
- **PTX / SASS**: portable vs. native GPU code (§9.2).
- **RPATH / RUNPATH**: per-library search paths baked into ELF files.
- **sdist**: source distribution; built on the user's machine.
- **SONAME**: a shared library's official runtime name, e.g. `libcublas.so.12`.
- **Tag (wheel)**: python-abi-platform triple in the filename.
- **venv**: isolated Python environment.
- **Wheel**: zip-format binary distribution; installing = unzipping.

---

### Further reading

- packaging.python.org: the official guides and specifications (wheel,
  `pyproject.toml`, version specifiers, platform tags).
- PEP 427 (wheel), 440 (versions), 508 (requirements), 517/518 (build
  system), 600 (manylinux), 621 (project metadata).
- `man ld.so`: the loader's own documentation of search order and
  `LD_DEBUG`.
- github.com/pypa/manylinux and github.com/pypa/auditwheel
- NVIDIA *CUDA Compatibility* guide (driver/runtime/minor-version rules) and
  the `nvcc` docs on `-gencode`/fatbinaries.
- `cibuildwheel`: the standard tool that automates manylinux builds for
  many Python versions in CI; worth knowing for projects with real extension
  modules.
