# DPP HIP Examples

Code examples to supplement the blog post on Data Parallel Primitives (DPP) in HIP.

- `dpp.hpp`contains a simple wrapper around the DPP intrinsics
- `print_dpp_intermediates.hip.cpp` shows intermediate values of DPP operations for wave reduction to illustrate how they work

This directory uses a minimal CMake setup:

- every `*.hip` / `*.hip.cpp` is built as its own executable
- each file is expected to have its own `main()`

## Build

```bash
cd dpp
cmake -S . -B build
cmake --build build -j
```
## Run

```bash
./build/print_dpp_intermediates
```
