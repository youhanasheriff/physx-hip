#!/bin/bash
# Builds the port's tests against a PhysX build.
#
# usage: build.sh [physx-bin-dir] [config]
# The binaries are written next to the PhysX libraries, so that libPhysXGpu_64.so is found.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PHYSX=$HERE/../../physx
CONFIG=${2:-checked}
BIN=${1:-$PHYSX/bin/linux.x86_64/$CONFIG}
case "$CONFIG" in
	debug)   DEFS="-DPX_DEBUG=1 -DPX_CHECKED=1" ;;
	checked) DEFS="-DNDEBUG -DPX_CHECKED=1" ;;
	profile) DEFS="-DNDEBUG -DPX_PROFILE=1" ;;
	release) DEFS="-DNDEBUG" ;;
	*) echo "unknown config $CONFIG" >&2; exit 2 ;;
esac
LIBS="-lPhysXExtensions_static_64 -lPhysX_static_64 -lPhysXPvdSDK_static_64 -lPhysXCommon_static_64 -lPhysXFoundation_static_64"
[ -f "$BIN/libPVDRuntime_static_64.a" ] && LIBS="$LIBS -lPVDRuntime_static_64"
${CXX:-clang++} -std=c++14 -O2 $DEFS -DPX_PHYSX_STATIC_LIB -DPX_SUPPORT_PVD=1 -DPX_GPU_BACKEND_HIP=1 \
	-I"$PHYSX/include" "$HERE/cpu_gpu_compare.cpp" -o "$BIN/cpu_gpu_compare" \
	-L"$BIN" -Wl,--start-group $LIBS -Wl,--end-group -lpthread -ldl -Wl,-rpath,'$ORIGIN'
echo "built $BIN/cpu_gpu_compare"
