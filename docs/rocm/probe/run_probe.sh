#!/usr/bin/env bash
# Compile-probe Humming's datatype/dequant layer for gfx950 with hipcc.
# shim/ provides cuda*.h -> hip*.h aliases and empty stubs for the NVIDIA-only
# PTX headers (TMA / mbarrier / WGMMA / tcgen05 / cp.async), so the remaining
# errors are what the "portable" C++ layer actually needs fixed.
set -euo pipefail
cd "$(dirname "$0")"
ARCH=${ARCH:-gfx950}
hipcc --offload-arch="$ARCH" -std=c++17 -O3 -ferror-limit=0 \
  -Ishim -I../../../humming/include -c probe_dequant.hip -o /dev/null 2>&1 |
  grep -E "error:" |
  sed -E 's/^.*include\/humming\///; s/:[0-9]+:[0-9]+:/:/' |
  sort | uniq -c | sort -rn || true
