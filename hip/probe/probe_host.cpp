// HIP capability probe: host side.
//
// This file is part of the HIP port and is not part of upstream PhysX.
//
// Drives the GPU through the same sequence of driver-style calls that PhysX's
// CudaContextManager makes (init, context, module load from memory, function lookup by
// name, launch through an argument-pointer array, streams, events, async copies and
// memsets), then checks every kernel result exactly.
//
// usage: probe_host <code-object-file> [device-ordinal]
// exit code 0 when every check passes.

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int gFailures = 0;
static int gChecks = 0;

#define CHECK(cond, ...)                                  \
	do {                                                  \
		++gChecks;                                        \
		if (!(cond)) {                                    \
			++gFailures;                                  \
			std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
			std::printf(__VA_ARGS__);                     \
			std::printf("\n");                            \
		}                                                 \
	} while (0)

#define HIP_OK(call)                                                                   \
	do {                                                                               \
		hipError_t e_ = (call);                                                        \
		++gChecks;                                                                     \
		if (e_ != hipSuccess) {                                                        \
			++gFailures;                                                               \
			std::printf("  FAIL %s:%d: %s -> %d (%s)\n", __FILE__, __LINE__, #call, int(e_), hipGetErrorString(e_)); \
		}                                                                              \
	} while (0)

struct ProbeDesc
{
	unsigned int count;
	float scale;
	unsigned long long* out;
};

static std::vector<char> readFile(const char* path)
{
	std::vector<char> data;
	if (FILE* f = std::fopen(path, "rb"))
	{
		std::fseek(f, 0, SEEK_END);
		const long size = std::ftell(f);
		std::fseek(f, 0, SEEK_SET);
		data.resize(size_t(size));
		if (std::fread(data.data(), 1, data.size(), f) != data.size())
			data.clear();
		std::fclose(f);
	}
	return data;
}

template <typename T>
static T* deviceAlloc(size_t count)
{
	void* p = nullptr;
	HIP_OK(hipMalloc(&p, count * sizeof(T)));
	return static_cast<T*>(p);
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::printf("usage: %s <code-object-file> [device-ordinal]\n", argv[0]);
		return 2;
	}
	const int ordinal = argc > 2 ? std::atoi(argv[2]) : 0;

	// ---- init, device and context, in the order CudaCtxMgr does it
	std::printf("[init]\n");
	HIP_OK(hipInit(0));
	int driverVersion = 0;
	HIP_OK(hipDriverGetVersion(&driverVersion));

	int deviceCount = 0;
	HIP_OK(hipGetDeviceCount(&deviceCount));
	if (deviceCount <= ordinal)
	{
		std::printf("  no HIP device at ordinal %d (device count %d)\n", ordinal, deviceCount);
		return 3;
	}

	hipDevice_t device = 0;
	HIP_OK(hipDeviceGet(&device, ordinal));

	hipCtx_t ctx = nullptr;
	HIP_OK(hipCtxCreate(&ctx, 0, device));

	char name[256] = {0};
	HIP_OK(hipDeviceGetName(name, sizeof(name), device));

	int sharedMemPerBlock = 0, clockRate = 0, capMajor = 0, capMinor = 0, integrated = 0, canMapHost = 0, cuCount = 0, maxThreads = 0, imageSupport = 0;
	HIP_OK(hipDeviceGetAttribute(&sharedMemPerBlock, hipDeviceAttributeMaxSharedMemoryPerBlock, device));
	HIP_OK(hipDeviceGetAttribute(&clockRate, hipDeviceAttributeClockRate, device));
	HIP_OK(hipDeviceGetAttribute(&capMajor, hipDeviceAttributeComputeCapabilityMajor, device));
	HIP_OK(hipDeviceGetAttribute(&capMinor, hipDeviceAttributeComputeCapabilityMinor, device));
	HIP_OK(hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, device));
	HIP_OK(hipDeviceGetAttribute(&canMapHost, hipDeviceAttributeCanMapHostMemory, device));
	HIP_OK(hipDeviceGetAttribute(&cuCount, hipDeviceAttributeMultiprocessorCount, device));
	HIP_OK(hipDeviceGetAttribute(&maxThreads, hipDeviceAttributeMaxThreadsPerBlock, device));
	HIP_OK(hipDeviceGetAttribute(&imageSupport, hipDeviceAttributeImageSupport, device));

	size_t totalMem = 0, freeMem = 0, totalMem2 = 0;
	HIP_OK(hipDeviceTotalMem(&totalMem, device));
	HIP_OK(hipMemGetInfo(&freeMem, &totalMem2));

	hipDeviceProp_t props;
	std::memset(&props, 0, sizeof(props));
	HIP_OK(hipGetDeviceProperties(&props, ordinal));

	std::printf("  device            : %s (%s)\n", name, props.gcnArchName);
	std::printf("  driver version    : %d\n", driverVersion);
	std::printf("  compute capability: %d.%d\n", capMajor, capMinor);
	std::printf("  warp size         : %d\n", props.warpSize);
	std::printf("  compute units     : %d, max threads/block %d, shared mem/block %d\n", cuCount, maxThreads, sharedMemPerBlock);
	std::printf("  memory            : total %.2f GiB, free %.2f GiB\n", double(totalMem) / (1 << 30), double(freeMem) / (1 << 30));
	std::printf("  integrated %d, can map host memory %d, image (texture) support %d\n", integrated, canMapHost, imageSupport);
	CHECK(props.warpSize == 32, "PhysX kernels need 32-lane wavefronts, device reports %d", props.warpSize);
	CHECK(maxThreads >= 1024, "PhysX launches 1024-thread blocks, device allows %d", maxThreads);

	int prioLow = 0, prioHigh = 0;
	HIP_OK(hipDeviceGetStreamPriorityRange(&prioLow, &prioHigh));

	// ---- module load from memory and function lookup by name
	std::printf("[module]\n");
	const std::vector<char> image = readFile(argv[1]);
	CHECK(!image.empty(), "could not read code object %s", argv[1]);
	if (image.empty())
		return 1;

	hipModule_t module = nullptr;
	HIP_OK(hipModuleLoadDataEx(&module, image.data(), 0, nullptr, nullptr));
	if (!module)
	{
		std::printf("  module did not load; nothing further can run\n");
		return 1;
	}

	hipFunction_t fWarp = nullptr, fShift = nullptr, fScan = nullptr, fAtomics = nullptr, fArgs = nullptr, fMissing = nullptr;
	HIP_OK(hipModuleGetFunction(&fWarp, module, "probeWarp"));
	HIP_OK(hipModuleGetFunction(&fShift, module, "probeShift"));
	HIP_OK(hipModuleGetFunction(&fScan, module, "probeSharedScan"));
	HIP_OK(hipModuleGetFunction(&fAtomics, module, "probeAtomics"));
	HIP_OK(hipModuleGetFunction(&fArgs, module, "probeArgs"));
	CHECK(hipModuleGetFunction(&fMissing, module, "noSuchKernel") != hipSuccess, "lookup of a missing kernel must fail");
	std::printf("  loaded %zu bytes, 5 kernels found by name\n", image.size());

	// ---- streams and events
	hipStream_t stream = nullptr, stream2 = nullptr;
	HIP_OK(hipStreamCreateWithPriority(&stream, 0, prioHigh));
	HIP_OK(hipStreamCreate(&stream2));
	hipEvent_t event = nullptr;
	HIP_OK(hipEventCreate(&event));

	// ---- probeWarp: reduction, vote, compaction
	std::printf("[warp shuffle / ballot / compaction]\n");
	{
		const unsigned int n = 100000;          // deliberately not a multiple of 32 or 256
		const unsigned int block = 256;
		const unsigned int grid = (n + block - 1) / block;
		const unsigned int padded = grid * block;
		const unsigned int warps = padded / 32;
		const float threshold = 0.5f;

		// page-locked host memory, as PhysX uses for its staging buffers
		float* hIn = nullptr;
		HIP_OK(hipHostMalloc(reinterpret_cast<void**>(&hIn), padded * sizeof(float), hipHostMallocDefault));
		unsigned int seed = 12345u;
		for (unsigned int i = 0; i < padded; ++i)
		{
			seed = seed * 1664525u + 1013904223u;
			hIn[i] = float(seed >> 8) / float(1u << 24); // [0,1)
		}

		float* dIn = deviceAlloc<float>(padded);
		float* dWarpSum = deviceAlloc<float>(warps);
		unsigned int* dWarpCount = deviceAlloc<unsigned int>(warps);
		unsigned int* dCompact = deviceAlloc<unsigned int>(padded);
		unsigned int* dCounter = deviceAlloc<unsigned int>(1);

		HIP_OK(hipMemcpyHtoDAsync(hipDeviceptr_t(dIn), hIn, padded * sizeof(float), stream));
		HIP_OK(hipMemsetD32Async(hipDeviceptr_t(dCounter), 0, 1, stream));
		HIP_OK(hipMemsetD32Async(hipDeviceptr_t(dCompact), 0xffffffff, padded, stream));

		unsigned int nArg = n;
		float thresholdArg = threshold;
		void* params[] = {&dIn, &nArg, &thresholdArg, &dWarpSum, &dWarpCount, &dCompact, &dCounter};
		HIP_OK(hipModuleLaunchKernel(fWarp, grid, 1, 1, block, 1, 1, 0, stream, params, nullptr));
		HIP_OK(hipEventRecord(event, stream));

		// a second stream waits on the first one's event before reading back
		std::vector<float> warpSum(warps);
		std::vector<unsigned int> warpCount(warps), compact(padded);
		unsigned int counter = 0;
		HIP_OK(hipStreamWaitEvent(stream2, event, 0));
		HIP_OK(hipMemcpyDtoHAsync(warpSum.data(), hipDeviceptr_t(dWarpSum), warps * sizeof(float), stream2));
		HIP_OK(hipMemcpyDtoHAsync(warpCount.data(), hipDeviceptr_t(dWarpCount), warps * sizeof(unsigned int), stream2));
		HIP_OK(hipMemcpyDtoHAsync(compact.data(), hipDeviceptr_t(dCompact), padded * sizeof(unsigned int), stream2));
		HIP_OK(hipMemcpyDtoHAsync(&counter, hipDeviceptr_t(dCounter), sizeof(counter), stream2));
		HIP_OK(hipStreamSynchronize(stream2));
		HIP_OK(hipEventSynchronize(event));
		CHECK(hipEventQuery(event) == hipSuccess, "event must be complete after synchronize");
		CHECK(hipStreamQuery(stream) == hipSuccess, "stream must be idle after synchronize");

		unsigned int expectedTotal = 0, badSum = 0, badCount = 0;
		std::vector<unsigned int> expectedKept;
		for (unsigned int w = 0; w < warps; ++w)
		{
			// same pairing order as the xor tree, so the float sum must match bit for bit
			float lanes[32];
			unsigned int count = 0;
			for (unsigned int l = 0; l < 32; ++l)
			{
				const unsigned int gid = w * 32 + l;
				lanes[l] = gid < n ? hIn[gid] : 0.0f;
				if (gid < n && hIn[gid] > threshold)
				{
					++count;
					expectedKept.push_back(gid);
				}
			}
			for (unsigned int step = 16; step >= 1; step >>= 1)
			{
				float next[32];
				for (unsigned int l = 0; l < 32; ++l)
					next[l] = lanes[l] + lanes[l ^ step];
				std::memcpy(lanes, next, sizeof(lanes));
			}
			if (warpSum[w] != lanes[0]) ++badSum;
			if (warpCount[w] != count) ++badCount;
			expectedTotal += count;
		}
		CHECK(badSum == 0, "%u of %u warp sums differ from the host reference", badSum, warps);
		CHECK(badCount == 0, "%u of %u warp ballots differ from the host reference", badCount, warps);
		CHECK(counter == expectedTotal, "compaction counter %u, expected %u", counter, expectedTotal);

		std::vector<unsigned int> kept(compact.begin(), compact.begin() + std::min<size_t>(counter, compact.size()));
		std::sort(kept.begin(), kept.end());
		CHECK(kept == expectedKept, "compacted set differs from the host reference (%zu vs %zu entries)", kept.size(), expectedKept.size());
		std::printf("  %u warps, %u elements kept\n", warps, counter);

		HIP_OK(hipFree(dIn)); HIP_OK(hipFree(dWarpSum)); HIP_OK(hipFree(dWarpCount)); HIP_OK(hipFree(dCompact)); HIP_OK(hipFree(dCounter));
		HIP_OK(hipHostFree(hIn));
	}

	// ---- probeShift: shifted shuffles, any/all votes
	std::printf("[shifted shuffle / any / all]\n");
	{
		const unsigned int block = 128, grid = 16, n = block * grid, warps = n / 32;
		std::vector<unsigned int> in(n), up(n), down(n), anyAll(warps);
		for (unsigned int i = 0; i < n; ++i)
			in[i] = (i / 32) % 3 == 0 ? i % 1000 : (i % 97 == 5 ? 5000 + i : i % 1000);

		unsigned int* dIn = deviceAlloc<unsigned int>(n);
		unsigned int* dUp = deviceAlloc<unsigned int>(n);
		unsigned int* dDown = deviceAlloc<unsigned int>(n);
		unsigned int* dAnyAll = deviceAlloc<unsigned int>(warps);
		HIP_OK(hipMemcpyHtoD(hipDeviceptr_t(dIn), in.data(), n * sizeof(unsigned int)));

		void* params[] = {&dIn, &dUp, &dDown, &dAnyAll};
		HIP_OK(hipModuleLaunchKernel(fShift, grid, 1, 1, block, 1, 1, 0, stream, params, nullptr));
		HIP_OK(hipStreamSynchronize(stream));
		HIP_OK(hipMemcpyDtoH(up.data(), hipDeviceptr_t(dUp), n * sizeof(unsigned int)));
		HIP_OK(hipMemcpyDtoH(down.data(), hipDeviceptr_t(dDown), n * sizeof(unsigned int)));
		HIP_OK(hipMemcpyDtoH(anyAll.data(), hipDeviceptr_t(dAnyAll), warps * sizeof(unsigned int)));

		unsigned int badUp = 0, badDown = 0, badVote = 0;
		for (unsigned int i = 0; i < n; ++i)
		{
			const unsigned int lane = i & 31;
			if (up[i] != (lane == 0 ? in[i] : in[i - 1])) ++badUp;
			if (down[i] != (lane == 31 ? in[i] : in[i + 1])) ++badDown;
		}
		for (unsigned int w = 0; w < warps; ++w)
		{
			bool anyBig = false;
			for (unsigned int l = 0; l < 32; ++l)
				anyBig = anyBig || in[w * 32 + l] >= 1000u;
			if (anyAll[w] != ((anyBig ? 1u : 0u) | (!anyBig ? 2u : 0u))) ++badVote;
		}
		CHECK(badUp == 0, "%u __shfl_up_sync results differ", badUp);
		CHECK(badDown == 0, "%u __shfl_down_sync results differ", badDown);
		CHECK(badVote == 0, "%u any/all votes differ", badVote);

		HIP_OK(hipFree(dIn)); HIP_OK(hipFree(dUp)); HIP_OK(hipFree(dDown)); HIP_OK(hipFree(dAnyAll));
	}

	// ---- probeSharedScan: shared memory and block barriers
	std::printf("[shared memory / __syncthreads]\n");
	{
		const unsigned int block = 1024, grid = 64, n = block * grid;
		std::vector<unsigned int> in(n), out(n);
		for (unsigned int i = 0; i < n; ++i)
			in[i] = (i * 2654435761u) >> 20;

		unsigned int* dIn = deviceAlloc<unsigned int>(n);
		unsigned int* dOut = deviceAlloc<unsigned int>(n);
		HIP_OK(hipMemcpyHtoD(hipDeviceptr_t(dIn), in.data(), n * sizeof(unsigned int)));

		void* params[] = {&dIn, &dOut};
		HIP_OK(hipModuleLaunchKernel(fScan, grid, 1, 1, block, 1, 1, 0, stream, params, nullptr));
		HIP_OK(hipStreamSynchronize(stream));
		HIP_OK(hipMemcpyDtoH(out.data(), hipDeviceptr_t(dOut), n * sizeof(unsigned int)));

		unsigned int bad = 0;
		for (unsigned int b = 0; b < grid; ++b)
		{
			unsigned int running = 0;
			for (unsigned int t = 0; t < block; ++t)
			{
				running += in[b * block + t];
				if (out[b * block + t] != running) ++bad;
			}
		}
		CHECK(bad == 0, "%u of %u prefix sums differ", bad, n);
		HIP_OK(hipFree(dIn)); HIP_OK(hipFree(dOut));
	}

	// ---- probeAtomics
	std::printf("[atomics]\n");
	{
		const unsigned int n = 1000003, block = 256, grid = (n + block - 1) / block;
		float* dSum = deviceAlloc<float>(1);
		int* dMax = deviceAlloc<int>(1);
		int* dMin = deviceAlloc<int>(1);
		unsigned int* dBits = deviceAlloc<unsigned int>(1);
		int* dCas = deviceAlloc<int>(1);
		unsigned int* dAnd = deviceAlloc<unsigned int>(1);

		const float zero = 0.0f;
		const int minInit = -1, maxInit = 0x7fffffff, casInit = -1;
		HIP_OK(hipMemcpyHtoD(hipDeviceptr_t(dSum), &zero, sizeof(zero)));
		HIP_OK(hipMemcpyHtoD(hipDeviceptr_t(dMax), &minInit, sizeof(int)));
		HIP_OK(hipMemcpyHtoD(hipDeviceptr_t(dMin), &maxInit, sizeof(int)));
		HIP_OK(hipMemcpyHtoD(hipDeviceptr_t(dCas), &casInit, sizeof(int)));
		HIP_OK(hipMemsetD32(hipDeviceptr_t(dBits), 0, 1));
		HIP_OK(hipMemsetD32(hipDeviceptr_t(dAnd), 0xffffffff, 1));

		unsigned int nArg = n;
		void* params[] = {&nArg, &dSum, &dMax, &dMin, &dBits, &dCas, &dAnd};
		HIP_OK(hipModuleLaunchKernel(fAtomics, grid, 1, 1, block, 1, 1, 0, stream, params, nullptr));
		HIP_OK(hipStreamSynchronize(stream));

		float sum = 0; int iMax = 0, iMin = 0, cas = 0; unsigned int bits = 0, andBits = 0;
		HIP_OK(hipMemcpyDtoH(&sum, hipDeviceptr_t(dSum), sizeof(sum)));
		HIP_OK(hipMemcpyDtoH(&iMax, hipDeviceptr_t(dMax), sizeof(int)));
		HIP_OK(hipMemcpyDtoH(&iMin, hipDeviceptr_t(dMin), sizeof(int)));
		HIP_OK(hipMemcpyDtoH(&bits, hipDeviceptr_t(dBits), sizeof(bits)));
		HIP_OK(hipMemcpyDtoH(&cas, hipDeviceptr_t(dCas), sizeof(int)));
		HIP_OK(hipMemcpyDtoH(&andBits, hipDeviceptr_t(dAnd), sizeof(andBits)));

		CHECK(sum == float(n), "atomicAdd(float) gave %.1f, expected %u", double(sum), n);
		CHECK(iMax == int(n - 1), "atomicMax gave %d, expected %u", iMax, n - 1);
		CHECK(iMin == 0, "atomicMin gave %d, expected 0", iMin);
		CHECK(bits == 0xffffffffu, "atomicOr gave %08x", bits);
		CHECK(andBits == 0xffffff00u, "atomicAnd gave %08x", andBits);
		CHECK(cas >= 0 && cas < int(n), "atomicCAS left %d, expected one thread index", cas);

		HIP_OK(hipFree(dSum)); HIP_OK(hipFree(dMax)); HIP_OK(hipFree(dMin)); HIP_OK(hipFree(dBits)); HIP_OK(hipFree(dCas)); HIP_OK(hipFree(dAnd));
	}

	// ---- probeArgs: mixed-size by-value arguments and a struct read through a pointer
	std::printf("[kernel argument marshalling]\n");
	{
		const unsigned int n = 4096, block = 64, grid = n / block;
		unsigned long long* dOut = deviceAlloc<unsigned long long>(n);
		ProbeDesc* dDesc = deviceAlloc<ProbeDesc>(1);
		HIP_OK(hipMemsetD8(hipDeviceptr_t(dOut), 0, n * sizeof(unsigned long long)));

		ProbeDesc desc;
		desc.count = n - 100; // the kernel must honour the smaller of the two bounds
		desc.scale = 0.5f;
		desc.out = dOut;
		HIP_OK(hipMemcpyHtoD(hipDeviceptr_t(dDesc), &desc, sizeof(desc)));

		bool flag = true;
		unsigned int nArg = n;
		float scale = 4.0f;
		double bias = 7.0;
		unsigned char tag = 0xA5;
		void* params[] = {&flag, &nArg, &scale, &bias, &dDesc, &tag};
		HIP_OK(hipModuleLaunchKernel(fArgs, grid, 1, 1, block, 1, 1, 0, stream, params, nullptr));
		HIP_OK(hipStreamSynchronize(stream));

		std::vector<unsigned long long> out(n);
		HIP_OK(hipMemcpyDtoH(out.data(), hipDeviceptr_t(dOut), n * sizeof(unsigned long long)));
		unsigned int bad = 0;
		for (unsigned int i = 0; i < n; ++i)
		{
			unsigned long long expected = 0;
			if (i < desc.count)
				expected = (unsigned long long)(double(i) * 4.0 * 0.5 + 7.0) | (0xA5ull << 40) | (1ull << 48);
			if (out[i] != expected) ++bad;
		}
		CHECK(bad == 0, "%u of %u argument-marshalling results differ", bad, n);
		HIP_OK(hipFree(dOut)); HIP_OK(hipFree(dDesc));
	}

	// ---- teardown in the order CudaCtxMgr does it
	HIP_OK(hipEventDestroy(event));
	HIP_OK(hipStreamDestroy(stream));
	HIP_OK(hipStreamDestroy(stream2));
	HIP_OK(hipModuleUnload(module));
	HIP_OK(hipCtxDestroy(ctx));

	std::printf("---\n%d checks, %d failed\n", gChecks, gFailures);
	return gFailures == 0 ? 0 : 1;
}
