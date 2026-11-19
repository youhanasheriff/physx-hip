#!/bin/bash
# Copyright (c) 2026 Youhana Sheriff
# SPDX-License-Identifier: Apache-2.0
#
# Builds and runs the HIP capability probe on one GPU.
#
# usage: run.sh [offload-arch] [device-ordinal]
# offload-arch defaults to the architecture rocminfo reports for the first GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ARCH=${1:-$(rocminfo | awk '/^ *Name: *gfx/ {print $2; exit}')}
ORDINAL=${2:-0}
[ -n "$ARCH" ] || { echo "no AMD GPU found by rocminfo; pass the offload arch explicitly" >&2; exit 2; }
mkdir -p build
hipcc --genco -x hip --offload-arch="$ARCH" -mno-wavefrontsize64 -O3 -I../compat/include \
	probe_kernels.cu -o "build/probe_kernels.$ARCH.co"
${CXX:-g++} -O2 -std=c++17 -Wall -Wno-deprecated-declarations -D__HIP_PLATFORM_AMD__ \
	probe_host.cpp -o build/probe_host -lamdhip64
exec ./build/probe_host "build/probe_kernels.$ARCH.co" "$ORDINAL"
