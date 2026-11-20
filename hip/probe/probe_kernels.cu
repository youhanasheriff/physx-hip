// Copyright (c) 2026 Youhana Sheriff
// SPDX-License-Identifier: Apache-2.0
//
// HIP capability probe: kernels written the way PhysX writes its CUDA kernels.
//
// This file is part of the HIP port and is not part of upstream PhysX.
//
// Each kernel exercises one family of features the PhysX GPU code depends on, with
// results the host can check exactly. See probe_host.cpp for the checks.

#include "PxHipDevice.h"

#define LOG2_WARP_SIZE 5
#define WARP_SIZE (1U << LOG2_WARP_SIZE)
#define FULL_MASK 0xffffffff

// Warp reduction, warp vote and warp-level stream compaction.
// Launched with blocks of 256 threads (8 warps).
extern "C" __global__ void probeWarp(
	const float* in, unsigned int n, float threshold,
	float* outWarpSum, unsigned int* outWarpCount,
	unsigned int* outCompact, unsigned int* compactCounter)
{
	const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
	const unsigned int warp = gid >> LOG2_WARP_SIZE;

	const float v = gid < n ? in[gid] : 0.0f;

	// tree reduction over the warp
	float sum = v;
	sum += __shfl_xor_sync(FULL_MASK, sum, 16);
	sum += __shfl_xor_sync(FULL_MASK, sum, 8);
	sum += __shfl_xor_sync(FULL_MASK, sum, 4);
	sum += __shfl_xor_sync(FULL_MASK, sum, 2);
	sum += __shfl_xor_sync(FULL_MASK, sum, 1);

	const bool keep = gid < n && v > threshold;
	const unsigned int ballot = __ballot_sync(FULL_MASK, keep);
	const unsigned int count = __popc(ballot);

	// lane 0 reserves output space for the whole warp, then broadcasts the base
	unsigned int base = 0;
	if (lane == 0)
	{
		outWarpSum[warp] = sum;
		outWarpCount[warp] = count;
		base = atomicAdd(compactCounter, count);
	}
	base = __shfl_sync(FULL_MASK, base, 0);

	if (keep)
	{
		const unsigned int offset = __popc(ballot & ((1u << lane) - 1u));
		outCompact[base + offset] = gid;
	}
}

// Shifted shuffles and a partial-warp vote.
// out[gid] = value of the previous lane (lane 0 keeps its own), then of the next lane.
extern "C" __global__ void probeShift(const unsigned int* in, unsigned int* outUp, unsigned int* outDown, unsigned int* outAnyAll)
{
	const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
	const unsigned int v = in[gid];

	outUp[gid] = __shfl_up_sync(FULL_MASK, v, 1);
	outDown[gid] = __shfl_down_sync(FULL_MASK, v, 1);

	const int anyBig = __any_sync(FULL_MASK, v >= 1000u);
	const int allSmall = __all_sync(FULL_MASK, v < 1000u);
	if (lane == 0)
		outAnyAll[gid >> LOG2_WARP_SIZE] = (anyBig ? 1u : 0u) | (allSmall ? 2u : 0u);
}

// Block-level inclusive prefix sum through shared memory and __syncthreads.
// Launched with blocks of 1024 threads.
extern "C" __global__ void __launch_bounds__(1024, 1) probeSharedScan(const unsigned int* in, unsigned int* out)
{
	__shared__ unsigned int sA[1024];
	__shared__ unsigned int sB[1024];

	const unsigned int tid = threadIdx.x;
	const unsigned int gid = blockIdx.x * blockDim.x + tid;

	sA[tid] = in[gid];
	__syncthreads();

	unsigned int* src = sA;
	unsigned int* dst = sB;
	for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1)
	{
		dst[tid] = tid >= stride ? src[tid] + src[tid - stride] : src[tid];
		__syncthreads();
		unsigned int* t = src; src = dst; dst = t;
	}
	out[gid] = src[tid];
}

// Atomics on the types PhysX uses them on.
extern "C" __global__ void probeAtomics(
	unsigned int n, float* fSum, int* iMax, int* iMin, unsigned int* bits, int* casSlot, unsigned int* andBits)
{
	const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= n)
		return;

	atomicAdd(fSum, 1.0f);
	atomicMax(iMax, int(gid));
	atomicMin(iMin, int(gid));
	atomicOr(bits, 1u << (gid & 31u));
	atomicAnd(andBits, ~(1u << (gid & 7u)));
	atomicCAS(casSlot, -1, int(gid));
}

// Mixed-size by-value arguments, passed the way PhysX passes them: an array of
// pointers to each argument, sized by the kernel signature.
struct ProbeDesc
{
	unsigned int count;
	float scale;
	unsigned long long* out;
};

extern "C" __global__ void probeArgs(bool flag, unsigned int n, float scale, double bias, const ProbeDesc* desc, unsigned char tag)
{
	const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= n || gid >= desc->count)
		return;

	const double value = double(gid) * double(scale) * double(desc->scale) + bias;
	const unsigned long long packed = (unsigned long long)(value) | ((unsigned long long)(tag) << 40) | ((unsigned long long)(flag ? 1 : 0) << 48);
	desc->out[gid] = packed;
}

// Writes a known value to every element after a short busy loop, which also makes it usable
// as a kernel that keeps the GPU busy. Used by check_idle_gpu_keeps_writes.cpp.
extern "C" __global__ void probeDelayedFill(unsigned int* out, unsigned int n, unsigned int value, unsigned int spin)
{
	const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= n)
		return;

	volatile unsigned int sink = 0;
	for (unsigned int i = 0; i < spin; ++i)
		sink = sink + i;
	out[gid] = value + gid;
}

// Counts the elements that hold the value probeDelayedFill wrote, as the GPU sees them.
extern "C" __global__ void probeCountFilled(const unsigned int* in, unsigned int n, unsigned int value, unsigned int* count)
{
	const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < n && in[gid] == value + gid)
		atomicAdd(count, 1u);
}
