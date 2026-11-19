#!/bin/bash
# Copyright (c) 2026 Youhana Sheriff
# SPDX-License-Identifier: Apache-2.0
#
# Compiles every PhysX kernel source (.cu) for an AMD GPU with the HIP compiler, device
# code only, and reports which files compile. Used to measure how much of the kernel
# code builds under HIP and to group the remaining errors.
#
# usage: kernel_census.sh <out-dir> [offload-arch]
# Writes <out-dir>/<name>.log and a linked code object <name>.co per file (loadable with
# hipModuleLoadData), and prints a summary.
set -u
OUT=${1:?usage: kernel_census.sh <out-dir> [offload-arch]}
ARCH=${2:-gfx1036}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PHYSX=$ROOT/physx
CLANG=${HIP_CLANG:-/usr/lib/llvm-21/bin/clang++}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)

INCS="-I$ROOT/hip/compat/include"
for d in include source/common/include source/common/src source/cudamanager/include \
	source/geomutils/include source/geomutils/src source/geomutils/src/common source/geomutils/src/contact \
	source/geomutils/src/convex source/geomutils/src/hf source/geomutils/src/mesh source/geomutils/src/pcm \
	source/gpuarticulation/include source/gpuarticulation/src/CUDA source/gpubroadphase/include \
	source/gpucommon/include source/gpucommon/src/CUDA source/gpunarrowphase/include source/gpunarrowphase/src/CUDA \
	source/gpusimulationcontroller/include source/gpusimulationcontroller/src/CUDA source/gpusolver/include \
	source/gpusolver/src/CUDA source/lowlevelaabb/include source/lowlevel/api/include \
	source/lowlevel/common/include/pipeline source/lowlevel/common/include/utils source/lowleveldynamics/include \
	source/lowleveldynamics/shared source/lowleveldynamics/src source/lowlevel/software/include \
	source/physxgpu/include source/physx/src/opensource/cudamanager/include; do
	INCS="$INCS -I$PHYSX/$d"
done
DEFS="-DNDEBUG -DPX_CHECKED=1 -DPX_NVTX=0 -DPX_PUBLIC_RELEASE=1 -DPX_SUPPORT_PVD=1 -DPX_SUPPORT_OMNI_PVD=1 -DPX_PHYSX_GPU_EXPORTS -D_CONSOLE"
FLAGS="-x hip --cuda-device-only --offload-arch=$ARCH -mno-wavefrontsize64 -std=c++14 -O3 -ffast-math -fno-rtti -fno-exceptions -ferror-limit=0 -w -include PxHipDevice.h"

# Files that hit a compiler defect at the default optimisation level, and the level that works.
opt_override() {
	case "$1" in
		convexCoreCollision) echo "-O1" ;; # LLVM 21 AMDGPU backend: "SOP2/SOPC instruction requires too many immediate constants" at -O2/-O3
	esac
}
compile_one() {
	src=$1; name=$(basename "$src" .cu)
	if $CLANG $FLAGS $(opt_override "$name") $DEFS $INCS "$src" -o "$OUT/$name.co" > "$OUT/$name.log" 2>&1; then
		echo "OK   $name"
	else
		echo "FAIL $name $(grep -c ' error: ' "$OUT/$name.log")"
	fi
}
export -f compile_one opt_override; export CLANG FLAGS DEFS INCS OUT
find "$PHYSX/source" -name '*.cu' | sort | xargs -P "${JOBS:-16}" -I{} bash -c 'compile_one {}' | sort > "$OUT/summary.txt"
cat "$OUT/summary.txt"
echo "---"
echo "ok=$(grep -c '^OK' "$OUT/summary.txt") fail=$(grep -c '^FAIL' "$OUT/summary.txt")"
