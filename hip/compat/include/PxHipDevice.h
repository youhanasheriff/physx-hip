// Device-side compatibility layer for building PhysX GPU kernels with HIP.
//
// This file is part of the HIP port and is not part of upstream PhysX.
//
// PhysX kernels are written for CUDA's 32-thread warps: WARP_SIZE is 32, FULL_MASK is
// 0xffffffff, and the *_sync warp builtins are called with 32-bit masks and expected to
// return 32-bit results. HIP ships *_sync builtins that insist on 64-bit masks and return
// 64-bit ballots, because Instinct GPUs run 64-lane wavefronts. This header switches those
// off and supplies CUDA-shaped 32-bit versions instead.
//
// Consequence: kernels built with this header are only correct on GPUs running 32-lane
// wavefronts (RDNA: gfx10xx, gfx11xx, gfx12xx). The build must pass -mno-wavefrontsize64
// and must not target gfx9xx.

#ifndef PX_HIP_DEVICE_H
#define PX_HIP_DEVICE_H

#ifndef HIP_DISABLE_WARP_SYNC_BUILTINS
#define HIP_DISABLE_WARP_SYNC_BUILTINS
#endif

#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__)

// In CUDA the mask names the threads that take part. On AMD hardware every lane of a
// wavefront that is active at the call site takes part, so the mask is only used to trim
// results, exactly as CUDA callers expect.

template <typename T>
__device__ __forceinline__ T __shfl_sync(unsigned int, T var, int srcLane, int width = 32)
{
	return __shfl(var, srcLane, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_up_sync(unsigned int, T var, unsigned int delta, int width = 32)
{
	return __shfl_up(var, delta, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_down_sync(unsigned int, T var, unsigned int delta, int width = 32)
{
	return __shfl_down(var, delta, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_xor_sync(unsigned int, T var, int laneMask, int width = 32)
{
	return __shfl_xor(var, laneMask, width);
}

__device__ __forceinline__ unsigned int __ballot_sync(unsigned int mask, int predicate)
{
	return static_cast<unsigned int>(__ballot(predicate)) & mask;
}

__device__ __forceinline__ int __any_sync(unsigned int mask, int predicate)
{
	return __ballot_sync(mask, predicate) != 0;
}

__device__ __forceinline__ int __all_sync(unsigned int mask, int predicate)
{
	const unsigned int active = static_cast<unsigned int>(__ballot(1)) & mask;
	return __ballot_sync(mask, predicate) == active;
}

__device__ __forceinline__ unsigned int __activemask()
{
	return static_cast<unsigned int>(__ballot(1));
}

__device__ __forceinline__ void __syncwarp(unsigned int = 0xffffffffu)
{
	__builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
	__builtin_amdgcn_wave_barrier();
	__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

#endif // __HIP_DEVICE_COMPILE__

#endif // PX_HIP_DEVICE_H
