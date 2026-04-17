# DPP HIP Examples

Code examples to supplement the blog post on Data Parallel Primitives (DPP) in HIP.

- `dpp.hpp` contains a simple wrapper around the DPP intrinsics
- `print_dpp_intermediates.hip` shows intermediate values of DPP operations for wave reduction to illustrate how they work.
- `bpermmute.hip` demonstrates the increasing latency of `ds_*` operations when multiple waves are issuing simultaneously.

## Build

```bash
cd dpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```
## Run

```bash
./build/print_dpp_intermediates
./build/bpermute_lcy
```
