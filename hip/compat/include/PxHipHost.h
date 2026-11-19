// Host-side compatibility layer for building PhysX GPU code with HIP.
//
// This file is part of the HIP port and is not part of upstream PhysX.
//
// PhysX's host code is written against the CUDA driver API (cu* functions, CU* types,
// CU_* constants). This header provides that vocabulary on top of the HIP runtime, for
// host translation units compiled by an ordinary C++ compiler.
//
// Handle types keep the exact struct tags PhysX forward declares in PxCudaTypes.h
// (CUstream_st and so on), so translation units that include this header and ones that
// only include PxCudaTypes.h agree on every type and mangled name, as they do in the CUDA
// build. The wrappers convert to HIP's handle types at the call.

#ifndef PX_HIP_HOST_H
#define PX_HIP_HOST_H

#if defined(__HIPCC__)
#error "PxHipHost.h is for host code; kernel sources get PxHipDevice.h"
#endif

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif
// HIP's vector types are kept out of the way; see PxHipVectorTypes.h.
#include "PxHipVectorRename.h"
#include <hip/hip_runtime_api.h>
#include <hip/driver_types.h>
#include <hip/texture_types.h>
#include "PxHipVectorUnrename.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

// CUDA's function markers, empty for host compilers (HIP defines the common ones already).
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __builtin_align__
#define __builtin_align__(a) __attribute__((aligned(a)))
#endif

// PxCudaTypes.h keys off this to know that the driver types are already declared.
// The value mirrors the CUDA toolkit PhysX 5.13 is written against (12.8).
#ifndef CUDA_VERSION
#define CUDA_VERSION 12080
#endif

// Calling convention marker of the CUDA driver API; empty on Linux.
#ifndef CUDAAPI
#define CUDAAPI
#endif

// ---- handle types (same declarations as PxCudaTypes.h)

typedef unsigned long long CUdeviceptr;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;
typedef struct CUgraphicsResource_st* CUgraphicsResource;

typedef hipArray_t CUarray;
typedef hipTextureObject_t CUtexObject;

// ---- results

typedef hipError_t CUresult;
#define CUDA_SUCCESS                         hipSuccess
#define CUDA_ERROR_INVALID_VALUE             hipErrorInvalidValue
#define CUDA_ERROR_OUT_OF_MEMORY             hipErrorOutOfMemory
#define CUDA_ERROR_NO_BINARY_FOR_GPU         hipErrorNoBinaryForGpu
#define CUDA_ERROR_NOT_READY                 hipErrorNotReady
#define CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES   hipErrorLaunchOutOfResources
#define CUDA_ERROR_UNKNOWN                   hipErrorUnknown

// ---- flags and enums

#define CU_MEMHOSTALLOC_PORTABLE       0x01
#define CU_MEMHOSTALLOC_DEVICEMAP      0x02
#define CU_MEMHOSTALLOC_WRITECOMBINED  0x04

#define CU_STREAM_DEFAULT        0x0
#define CU_STREAM_NON_BLOCKING   0x1

#define CU_EVENT_DEFAULT          0x0
#define CU_EVENT_BLOCKING_SYNC    0x1
#define CU_EVENT_DISABLE_TIMING   0x2

// Context creation flags. HIP contexts take no scheduling or mapping flags.
#define CU_CTX_SCHED_BLOCKING_SYNC   0x04
#define CU_CTX_MAP_HOST              0x08
#define CU_CTX_LMEM_RESIZE_TO_MAX    0x10

typedef hipDeviceAttribute_t CUdevice_attribute;
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                  hipDeviceAttributeMaxThreadsPerBlock
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK            hipDeviceAttributeMaxSharedMemoryPerBlock
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR   hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE                             hipDeviceAttributeClockRate
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR               hipDeviceAttributeComputeCapabilityMajor
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR               hipDeviceAttributeComputeCapabilityMinor
#define CU_DEVICE_ATTRIBUTE_INTEGRATED                             hipDeviceAttributeIntegrated
#define CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                    hipDeviceAttributeCanMapHostMemory
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                   hipDeviceAttributeMultiprocessorCount

typedef hipJitOption CUjit_option;

// ---- arrays and textures (used for SDF collision)

typedef hipArray_Format CUarray_format;
#define CU_AD_FORMAT_UNSIGNED_INT8    HIP_AD_FORMAT_UNSIGNED_INT8
#define CU_AD_FORMAT_UNSIGNED_INT16   HIP_AD_FORMAT_UNSIGNED_INT16
#define CU_AD_FORMAT_FLOAT            HIP_AD_FORMAT_FLOAT

#define CU_TR_ADDRESS_MODE_CLAMP   HIP_TR_ADDRESS_MODE_CLAMP
#define CU_TR_FILTER_MODE_LINEAR   HIP_TR_FILTER_MODE_LINEAR
#define CU_RESOURCE_TYPE_ARRAY     HIP_RESOURCE_TYPE_ARRAY
#define CU_MEMORYTYPE_HOST         hipMemoryTypeHost
#define CU_MEMORYTYPE_ARRAY        hipMemoryTypeArray

typedef HIP_ARRAY3D_DESCRIPTOR CUDA_ARRAY3D_DESCRIPTOR;
typedef HIP_ARRAY3D_DESCRIPTOR CUDA_ARRAY3D_DESCRIPTOR_st;
typedef HIP_MEMCPY3D CUDA_MEMCPY3D;
typedef HIP_RESOURCE_DESC CUDA_RESOURCE_DESC;
typedef HIP_TEXTURE_DESC CUDA_TEXTURE_DESC;

// ---- handle conversions

namespace pxhip
{
	inline hipCtx_t ctx(CUcontext c) { return reinterpret_cast<hipCtx_t>(c); }
	inline hipModule_t mod(CUmodule m) { return reinterpret_cast<hipModule_t>(m); }
	inline hipFunction_t fn(CUfunction f) { return reinterpret_cast<hipFunction_t>(f); }
	inline hipStream_t stream(CUstream s) { return reinterpret_cast<hipStream_t>(s); }
	inline hipEvent_t event(CUevent e) { return reinterpret_cast<hipEvent_t>(e); }
	inline hipDeviceptr_t ptr(CUdeviceptr p) { return reinterpret_cast<hipDeviceptr_t>(p); }
}

// ---- driver API

inline CUresult cuInit(unsigned int flags) { return hipInit(flags); }
inline CUresult cuDriverGetVersion(int* version) { return hipDriverGetVersion(version); }

inline CUresult cuDeviceGet(CUdevice* device, int ordinal) { return hipDeviceGet(device, ordinal); }
inline CUresult cuDeviceGetCount(int* count) { return hipGetDeviceCount(count); }
inline CUresult cuDeviceGetName(char* name, int len, CUdevice device) { return hipDeviceGetName(name, len, device); }
inline CUresult cuDeviceGetAttribute(int* value, CUdevice_attribute attribute, CUdevice device) { return hipDeviceGetAttribute(value, attribute, device); }
inline CUresult cuDeviceTotalMem(size_t* bytes, CUdevice device) { return hipDeviceTotalMem(bytes, device); }

// HIP's context functions are deprecated but functional: a context stands for a device,
// and push/pop select the calling thread's current device, which is how PhysX uses them.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
inline CUresult cuCtxCreate(CUcontext* ctx, unsigned int /*flags*/, CUdevice device) { return hipCtxCreate(reinterpret_cast<hipCtx_t*>(ctx), 0, device); }
inline CUresult cuCtxDestroy(CUcontext ctx) { return hipCtxDestroy(pxhip::ctx(ctx)); }
inline CUresult cuCtxPushCurrent(CUcontext ctx) { return hipCtxPushCurrent(pxhip::ctx(ctx)); }
inline CUresult cuCtxPopCurrent(CUcontext* ctx) { return hipCtxPopCurrent(reinterpret_cast<hipCtx_t*>(ctx)); }
inline CUresult cuCtxGetDevice(CUdevice* device) { return hipCtxGetDevice(device); }
inline CUresult cuCtxSynchronize() { return hipCtxSynchronize(); }
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
inline CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) { return hipDeviceGetStreamPriorityRange(leastPriority, greatestPriority); }

inline CUresult cuMemGetInfo(size_t* freeBytes, size_t* totalBytes) { return hipMemGetInfo(freeBytes, totalBytes); }
inline CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytes)
{
	void* p = NULL;
	const CUresult r = hipMalloc(&p, bytes);
	*dptr = reinterpret_cast<CUdeviceptr>(p);
	return r;
}
inline CUresult cuMemFree(CUdeviceptr dptr) { return hipFree(pxhip::ptr(dptr)); }
// The CU_MEMHOSTALLOC_* flag values equal HIP's hipHostMalloc* flag values.
inline CUresult cuMemHostAlloc(void** pp, size_t bytes, unsigned int flags) { return hipHostMalloc(pp, bytes, flags); }
inline CUresult cuMemFreeHost(void* p) { return hipHostFree(p); }
inline CUresult cuMemHostGetDevicePointer(CUdeviceptr* dptr, void* p, unsigned int flags)
{
	void* d = NULL;
	const CUresult r = hipHostGetDevicePointer(&d, p, flags);
	*dptr = reinterpret_cast<CUdeviceptr>(d);
	return r;
}

inline CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues)
{
	return hipModuleLoadDataEx(reinterpret_cast<hipModule_t*>(module), image, numOptions, options, optionValues);
}
inline CUresult cuModuleGetFunction(CUfunction* function, CUmodule module, const char* name) { return hipModuleGetFunction(reinterpret_cast<hipFunction_t*>(function), pxhip::mod(module), name); }
inline CUresult cuModuleUnload(CUmodule module) { return hipModuleUnload(pxhip::mod(module)); }

inline CUresult cuStreamCreate(CUstream* stream, unsigned int flags) { return hipStreamCreateWithFlags(reinterpret_cast<hipStream_t*>(stream), flags); }
inline CUresult cuStreamCreateWithPriority(CUstream* stream, unsigned int flags, int priority) { return hipStreamCreateWithPriority(reinterpret_cast<hipStream_t*>(stream), flags, priority); }
inline CUresult cuStreamDestroy(CUstream stream) { return hipStreamDestroy(pxhip::stream(stream)); }
inline CUresult cuStreamQuery(CUstream stream) { return hipStreamQuery(pxhip::stream(stream)); }
inline CUresult cuStreamSynchronize(CUstream stream) { return hipStreamSynchronize(pxhip::stream(stream)); }
inline CUresult cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags) { return hipStreamWaitEvent(pxhip::stream(stream), pxhip::event(event), flags); }

inline CUresult cuEventCreate(CUevent* event, unsigned int flags) { return hipEventCreateWithFlags(reinterpret_cast<hipEvent_t*>(event), flags); }
inline CUresult cuEventRecord(CUevent event, CUstream stream) { return hipEventRecord(pxhip::event(event), pxhip::stream(stream)); }
inline CUresult cuEventQuery(CUevent event) { return hipEventQuery(pxhip::event(event)); }
inline CUresult cuEventSynchronize(CUevent event) { return hipEventSynchronize(pxhip::event(event)); }
inline CUresult cuEventDestroy(CUevent event) { return hipEventDestroy(pxhip::event(event)); }

inline CUresult cuLaunchKernel(CUfunction f, unsigned int gridX, unsigned int gridY, unsigned int gridZ,
	unsigned int blockX, unsigned int blockY, unsigned int blockZ, unsigned int sharedMemBytes,
	CUstream stream, void** kernelParams, void** extra)
{
	return hipModuleLaunchKernel(pxhip::fn(f), gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, pxhip::stream(stream), kernelParams, extra);
}

inline CUresult cuMemcpyDtoH(void* dst, CUdeviceptr src, size_t bytes) { return hipMemcpyDtoH(dst, pxhip::ptr(src), bytes); }
inline CUresult cuMemcpyDtoHAsync(void* dst, CUdeviceptr src, size_t bytes, CUstream stream) { return hipMemcpyDtoHAsync(dst, pxhip::ptr(src), bytes, pxhip::stream(stream)); }
inline CUresult cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t bytes) { return hipMemcpyHtoD(pxhip::ptr(dst), const_cast<void*>(src), bytes); }
inline CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void* src, size_t bytes, CUstream stream) { return hipMemcpyHtoDAsync(pxhip::ptr(dst), const_cast<void*>(src), bytes, pxhip::stream(stream)); }
inline CUresult cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t bytes) { return hipMemcpyDtoD(pxhip::ptr(dst), pxhip::ptr(src), bytes); }
inline CUresult cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream) { return hipMemcpyDtoDAsync(pxhip::ptr(dst), pxhip::ptr(src), bytes, pxhip::stream(stream)); }
// PhysX only copies between contexts of the same device, which is a device-to-device copy.
inline CUresult cuMemcpyPeerAsync(CUdeviceptr dst, CUcontext, CUdeviceptr src, CUcontext, size_t bytes, CUstream stream) { return hipMemcpyDtoDAsync(pxhip::ptr(dst), pxhip::ptr(src), bytes, pxhip::stream(stream)); }

inline CUresult cuMemsetD8(CUdeviceptr dst, unsigned char value, size_t count) { return hipMemsetD8(pxhip::ptr(dst), value, count); }
inline CUresult cuMemsetD16(CUdeviceptr dst, unsigned short value, size_t count) { return hipMemsetD16(pxhip::ptr(dst), value, count); }
inline CUresult cuMemsetD32(CUdeviceptr dst, unsigned int value, size_t count) { return hipMemsetD32(pxhip::ptr(dst), int(value), count); }
inline CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char value, size_t count, CUstream stream) { return hipMemsetD8Async(pxhip::ptr(dst), value, count, pxhip::stream(stream)); }
inline CUresult cuMemsetD32Async(CUdeviceptr dst, unsigned int value, size_t count, CUstream stream) { return hipMemsetD32Async(pxhip::ptr(dst), int(value), count, pxhip::stream(stream)); }

inline CUresult cuArray3DCreate(CUarray* array, const CUDA_ARRAY3D_DESCRIPTOR* desc) { return hipArray3DCreate(array, desc); }
inline CUresult cuArrayDestroy(CUarray array) { return hipArrayDestroy(array); }
inline CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* copy, CUstream stream) { return hipDrvMemcpy3DAsync(copy, pxhip::stream(stream)); }
inline CUresult cuTexObjectCreate(CUtexObject* texture, const CUDA_RESOURCE_DESC* resDesc, const CUDA_TEXTURE_DESC* texDesc, const void* /*resViewDesc*/)
{
	return hipTexObjectCreate(texture, resDesc, texDesc, NULL);
}
inline CUresult cuTexObjectDestroy(CUtexObject texture) { return hipTexObjectDestroy(texture); }

inline CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction f, int blockSize, size_t dynamicSMemSize)
{
	return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, pxhip::fn(f), blockSize, dynamicSMemSize);
}

#endif // PX_HIP_HOST_H
