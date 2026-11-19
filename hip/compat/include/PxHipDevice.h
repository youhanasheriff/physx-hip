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

// HIP's vector types are kept out of the way; see PxHipVectorTypes.h.
#include "PxHipVectorRename.h"
#include <hip/hip_runtime.h>
#include "PxHipVectorUnrename.h"
#include "PxHipVectorTypes.h"

#if defined(__HIP_DEVICE_COMPILE__)

// The macro is deprecated in recent clang; while it exists, use it to reject wave64 builds.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-pragma"
#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "PhysX GPU kernels assume 32-thread warps; build for a wave32 target with -mno-wavefrontsize64"
#endif
#pragma clang diagnostic pop

// PhysX selects its device code paths with __CUDA_ARCH__. Report a recent architecture
// level so the modern paths (warp shuffles, 64-bit atomics) are the ones compiled.
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 800
#endif

// CUDA driver handle for a texture object, as stored in PhysX's shared structs.
typedef hipTextureObject_t CUtexObject;

// CUDA's struct alignment keyword.
#ifndef __builtin_align__
#define __builtin_align__(a) __attribute__((aligned(a)))
#endif

// CUDA's cache-hinted loads and stores (sm_35 intrinsics). The hints steer NVIDIA's cache
// hierarchy and have no AMD counterpart, so these are plain loads and stores. HIP declares
// __ldg for a fixed list of scalar and vector types; the template covers every other type.
template <typename T> __device__ __forceinline__ T __ldg(const T* p) { return *p; }
template <typename T> __device__ __forceinline__ T __ldca(const T* p) { return *p; }
template <typename T> __device__ __forceinline__ T __ldcg(const T* p) { return *p; }
template <typename T> __device__ __forceinline__ T __ldcs(const T* p) { return *p; }
template <typename T> __device__ __forceinline__ T __ldlu(const T* p) { return *p; }
template <typename T> __device__ __forceinline__ T __ldcv(const T* p) { return *p; }
template <typename T> __device__ __forceinline__ void __stwb(T* p, const T& v) { *p = v; }
template <typename T> __device__ __forceinline__ void __stcg(T* p, const T& v) { *p = v; }
template <typename T> __device__ __forceinline__ void __stcs(T* p, const T& v) { *p = v; }
template <typename T> __device__ __forceinline__ void __stwt(T* p, const T& v) { *p = v; }

// On Linux PhysX's 64-bit integers are 'long', which is a different type from the
// 'long long' HIP declares these for; without exact overloads the calls are ambiguous.
__device__ __forceinline__ int __ffsll(unsigned long v) { return __ffsll(static_cast<unsigned long long>(v)); }
__device__ __forceinline__ int __ffsll(long v) { return __ffsll(static_cast<unsigned long long>(v)); }

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
