// Compatibility header for building PhysX GPU code with HIP.
// This file is part of the HIP port and is not part of upstream PhysX.
//
// CUDA-compatible vector types (float4, uint4, ...) and their make_* constructors.
//
// PhysX shares these types between host code and kernels and relies on CUDA's layout:
// plain aggregates, with two- and four-component types aligned to their full size (capped
// at 16 bytes) and one- and three-component types naturally aligned. HIP's own vector
// types are class templates with built-in operators, which clash with the operators PhysX
// defines in cutil_math.h, so the port uses these plain structs instead.

#ifndef PX_HIP_VECTOR_TYPES_H
#define PX_HIP_VECTOR_TYPES_H

#if defined(__HIPCC__)
#define PX_HIP_VEC_FN __host__ __device__ static inline
#else
#define PX_HIP_VEC_FN static inline
#endif

struct char1 { signed char x; };
struct alignas(2) char2 { signed char x; signed char y; };
struct char3 { signed char x; signed char y; signed char z; };
struct alignas(4) char4 { signed char x; signed char y; signed char z; signed char w; };

struct uchar1 { unsigned char x; };
struct alignas(2) uchar2 { unsigned char x; unsigned char y; };
struct uchar3 { unsigned char x; unsigned char y; unsigned char z; };
struct alignas(4) uchar4 { unsigned char x; unsigned char y; unsigned char z; unsigned char w; };

struct short1 { short x; };
struct alignas(4) short2 { short x; short y; };
struct short3 { short x; short y; short z; };
struct alignas(8) short4 { short x; short y; short z; short w; };

struct ushort1 { unsigned short x; };
struct alignas(4) ushort2 { unsigned short x; unsigned short y; };
struct ushort3 { unsigned short x; unsigned short y; unsigned short z; };
struct alignas(8) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; };

struct int1 { int x; };
struct alignas(8) int2 { int x; int y; };
struct int3 { int x; int y; int z; };
struct alignas(16) int4 { int x; int y; int z; int w; };

struct uint1 { unsigned int x; };
struct alignas(8) uint2 { unsigned int x; unsigned int y; };
struct uint3 { unsigned int x; unsigned int y; unsigned int z; };
struct alignas(16) uint4 { unsigned int x; unsigned int y; unsigned int z; unsigned int w; };

struct long1 { long int x; };
struct alignas(16) long2 { long int x; long int y; };
struct long3 { long int x; long int y; long int z; };
struct alignas(16) long4 { long int x; long int y; long int z; long int w; };

struct ulong1 { unsigned long int x; };
struct alignas(16) ulong2 { unsigned long int x; unsigned long int y; };
struct ulong3 { unsigned long int x; unsigned long int y; unsigned long int z; };
struct alignas(16) ulong4 { unsigned long int x; unsigned long int y; unsigned long int z; unsigned long int w; };

struct longlong1 { long long int x; };
struct alignas(16) longlong2 { long long int x; long long int y; };
struct longlong3 { long long int x; long long int y; long long int z; };
struct alignas(16) longlong4 { long long int x; long long int y; long long int z; long long int w; };

struct ulonglong1 { unsigned long long int x; };
struct alignas(16) ulonglong2 { unsigned long long int x; unsigned long long int y; };
struct ulonglong3 { unsigned long long int x; unsigned long long int y; unsigned long long int z; };
struct alignas(16) ulonglong4 { unsigned long long int x; unsigned long long int y; unsigned long long int z; unsigned long long int w; };

struct float1 { float x; };
struct alignas(8) float2 { float x; float y; };
struct float3 { float x; float y; float z; };
struct alignas(16) float4 { float x; float y; float z; float w; };

struct double1 { double x; };
struct alignas(16) double2 { double x; double y; };
struct double3 { double x; double y; double z; };
struct alignas(16) double4 { double x; double y; double z; double w; };

PX_HIP_VEC_FN char1 make_char1(signed char x) { char1 t; t.x = x; return t; }
PX_HIP_VEC_FN char2 make_char2(signed char x, signed char y) { char2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN char3 make_char3(signed char x, signed char y, signed char z) { char3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN char4 make_char4(signed char x, signed char y, signed char z, signed char w) { char4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PX_HIP_VEC_FN uchar1 make_uchar1(unsigned char x) { uchar1 t; t.x = x; return t; }
PX_HIP_VEC_FN uchar2 make_uchar2(unsigned char x, unsigned char y) { uchar2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) { uchar3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) { uchar4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PX_HIP_VEC_FN short1 make_short1(short x) { short1 t; t.x = x; return t; }
PX_HIP_VEC_FN short2 make_short2(short x, short y) { short2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN short3 make_short3(short x, short y, short z) { short3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN short4 make_short4(short x, short y, short z, short w) { short4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PX_HIP_VEC_FN ushort1 make_ushort1(unsigned short x) { ushort1 t; t.x = x; return t; }
PX_HIP_VEC_FN ushort2 make_ushort2(unsigned short x, unsigned short y) { ushort2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) { ushort3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) { ushort4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PX_HIP_VEC_FN int1 make_int1(int x) { int1 t; t.x = x; return t; }
PX_HIP_VEC_FN int2 make_int2(int x, int y) { int2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN int3 make_int3(int x, int y, int z) { int3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN int4 make_int4(int x, int y, int z, int w) { int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PX_HIP_VEC_FN uint1 make_uint1(unsigned int x) { uint1 t; t.x = x; return t; }
PX_HIP_VEC_FN uint2 make_uint2(unsigned int x, unsigned int y) { uint2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { uint4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PX_HIP_VEC_FN long1 make_long1(long int x) { long1 t; t.x = x; return t; }
PX_HIP_VEC_FN long2 make_long2(long int x, long int y) { long2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN long3 make_long3(long int x, long int y, long int z) { long3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN long4 make_long4(long int x, long int y, long int z, long int w) { long4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PX_HIP_VEC_FN ulong1 make_ulong1(unsigned long int x) { ulong1 t; t.x = x; return t; }
PX_HIP_VEC_FN ulong2 make_ulong2(unsigned long int x, unsigned long int y) { ulong2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z) { ulong3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN ulong4 make_ulong4(unsigned long int x, unsigned long int y, unsigned long int z, unsigned long int w) { ulong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PX_HIP_VEC_FN longlong1 make_longlong1(long long int x) { longlong1 t; t.x = x; return t; }
PX_HIP_VEC_FN longlong2 make_longlong2(long long int x, long long int y) { longlong2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN longlong3 make_longlong3(long long int x, long long int y, long long int z) { longlong3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN longlong4 make_longlong4(long long int x, long long int y, long long int z, long long int w) { longlong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PX_HIP_VEC_FN ulonglong1 make_ulonglong1(unsigned long long int x) { ulonglong1 t; t.x = x; return t; }
PX_HIP_VEC_FN ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y) { ulonglong2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN ulonglong3 make_ulonglong3(unsigned long long int x, unsigned long long int y, unsigned long long int z) { ulonglong3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN ulonglong4 make_ulonglong4(unsigned long long int x, unsigned long long int y, unsigned long long int z, unsigned long long int w) { ulonglong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PX_HIP_VEC_FN float1 make_float1(float x) { float1 t; t.x = x; return t; }
PX_HIP_VEC_FN float2 make_float2(float x, float y) { float2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN float3 make_float3(float x, float y, float z) { float3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN float4 make_float4(float x, float y, float z, float w) { float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PX_HIP_VEC_FN double1 make_double1(double x) { double1 t; t.x = x; return t; }
PX_HIP_VEC_FN double2 make_double2(double x, double y) { double2 t; t.x = x; t.y = y; return t; }
PX_HIP_VEC_FN double3 make_double3(double x, double y, double z) { double3 t; t.x = x; t.y = y; t.z = z; return t; }
PX_HIP_VEC_FN double4 make_double4(double x, double y, double z, double w) { double4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

// Layout checks: sizes and alignments PhysX's shared structs depend on.
static_assert(sizeof(char1) == 1 && alignof(char1) == 1, "char1 layout");
static_assert(sizeof(char2) == 2 && alignof(char2) == 2, "char2 layout");
static_assert(sizeof(char3) == 3 && alignof(char3) == 1, "char3 layout");
static_assert(sizeof(char4) == 4 && alignof(char4) == 4, "char4 layout");
static_assert(sizeof(uchar1) == 1 && alignof(uchar1) == 1, "uchar1 layout");
static_assert(sizeof(uchar2) == 2 && alignof(uchar2) == 2, "uchar2 layout");
static_assert(sizeof(uchar3) == 3 && alignof(uchar3) == 1, "uchar3 layout");
static_assert(sizeof(uchar4) == 4 && alignof(uchar4) == 4, "uchar4 layout");
static_assert(sizeof(short1) == 2 && alignof(short1) == 2, "short1 layout");
static_assert(sizeof(short2) == 4 && alignof(short2) == 4, "short2 layout");
static_assert(sizeof(short3) == 6 && alignof(short3) == 2, "short3 layout");
static_assert(sizeof(short4) == 8 && alignof(short4) == 8, "short4 layout");
static_assert(sizeof(ushort1) == 2 && alignof(ushort1) == 2, "ushort1 layout");
static_assert(sizeof(ushort2) == 4 && alignof(ushort2) == 4, "ushort2 layout");
static_assert(sizeof(ushort3) == 6 && alignof(ushort3) == 2, "ushort3 layout");
static_assert(sizeof(ushort4) == 8 && alignof(ushort4) == 8, "ushort4 layout");
static_assert(sizeof(int1) == 4 && alignof(int1) == 4, "int1 layout");
static_assert(sizeof(int2) == 8 && alignof(int2) == 8, "int2 layout");
static_assert(sizeof(int3) == 12 && alignof(int3) == 4, "int3 layout");
static_assert(sizeof(int4) == 16 && alignof(int4) == 16, "int4 layout");
static_assert(sizeof(uint1) == 4 && alignof(uint1) == 4, "uint1 layout");
static_assert(sizeof(uint2) == 8 && alignof(uint2) == 8, "uint2 layout");
static_assert(sizeof(uint3) == 12 && alignof(uint3) == 4, "uint3 layout");
static_assert(sizeof(uint4) == 16 && alignof(uint4) == 16, "uint4 layout");
static_assert(sizeof(long1) == 8 && alignof(long1) == 8, "long1 layout");
static_assert(sizeof(long2) == 16 && alignof(long2) == 16, "long2 layout");
static_assert(sizeof(long3) == 24 && alignof(long3) == 8, "long3 layout");
static_assert(sizeof(long4) == 32 && alignof(long4) == 16, "long4 layout");
static_assert(sizeof(ulong1) == 8 && alignof(ulong1) == 8, "ulong1 layout");
static_assert(sizeof(ulong2) == 16 && alignof(ulong2) == 16, "ulong2 layout");
static_assert(sizeof(ulong3) == 24 && alignof(ulong3) == 8, "ulong3 layout");
static_assert(sizeof(ulong4) == 32 && alignof(ulong4) == 16, "ulong4 layout");
static_assert(sizeof(longlong1) == 8 && alignof(longlong1) == 8, "longlong1 layout");
static_assert(sizeof(longlong2) == 16 && alignof(longlong2) == 16, "longlong2 layout");
static_assert(sizeof(longlong3) == 24 && alignof(longlong3) == 8, "longlong3 layout");
static_assert(sizeof(longlong4) == 32 && alignof(longlong4) == 16, "longlong4 layout");
static_assert(sizeof(ulonglong1) == 8 && alignof(ulonglong1) == 8, "ulonglong1 layout");
static_assert(sizeof(ulonglong2) == 16 && alignof(ulonglong2) == 16, "ulonglong2 layout");
static_assert(sizeof(ulonglong3) == 24 && alignof(ulonglong3) == 8, "ulonglong3 layout");
static_assert(sizeof(ulonglong4) == 32 && alignof(ulonglong4) == 16, "ulonglong4 layout");
static_assert(sizeof(float1) == 4 && alignof(float1) == 4, "float1 layout");
static_assert(sizeof(float2) == 8 && alignof(float2) == 8, "float2 layout");
static_assert(sizeof(float3) == 12 && alignof(float3) == 4, "float3 layout");
static_assert(sizeof(float4) == 16 && alignof(float4) == 16, "float4 layout");
static_assert(sizeof(double1) == 8 && alignof(double1) == 8, "double1 layout");
static_assert(sizeof(double2) == 16 && alignof(double2) == 16, "double2 layout");
static_assert(sizeof(double3) == 24 && alignof(double3) == 8, "double3 layout");
static_assert(sizeof(double4) == 32 && alignof(double4) == 16, "double4 layout");

#endif // PX_HIP_VECTOR_TYPES_H
