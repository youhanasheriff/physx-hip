// Copyright (c) 2026 Youhana Sheriff
// SPDX-License-Identifier: Apache-2.0
//
// Checks that what a kernel wrote is still there when the host does not wait for it straight away.
//
// Each round launches a kernel that fills a buffer, leaves the host idle for a while, and only then
// waits for the stream and reads the buffer back, both with a copy to the host and with a second kernel.
// With the HIP runtime's default setting (AMD_OPT_FLUSH=1) the writes stay in the GPU's caches until that
// wait. On the integrated GPU this port was developed on (gfx1036) they are lost when the pause is about
// a millisecond or longer, unless another stream keeps the GPU busy during the pause. With
// AMD_OPT_FLUSH=0, which the PhysX GPU library sets for itself, nothing is lost.
//
// This file is part of the HIP port and is not part of upstream PhysX.
// build: g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ check_idle_gpu_keeps_writes.cpp -o build/check_idle_gpu_keeps_writes -lamdhip64
// usage: [AMD_OPT_FLUSH=0|1] build/check_idle_gpu_keeps_writes build/probe_kernels.<arch>.co [rounds]
// exit code 0 when no round lost anything, 1 otherwise.

#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unistd.h>
static std::vector<char> readFile(const char* p){std::vector<char> d; FILE* f=fopen(p,"rb"); if(!f){perror(p); exit(2);} fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET); d.resize(s); if(fread(d.data(),1,s,f)!=size_t(s)) exit(2); fclose(f); return d;}
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("%s failed: %s\n", #x, hipGetErrorString(e_)); return 2; } } while (0)

int main(int argc, char** argv)
{
	if (argc < 2) { printf("usage: %s <kernels.co> [rounds]\n", argv[0]); return 2; }
	const int rounds = argc > 2 ? atoi(argv[2]) : 20;
	const char* flush = getenv("AMD_OPT_FLUSH");

	CHECK(hipInit(0));
	auto image = readFile(argv[1]);
	hipModule_t module; CHECK(hipModuleLoadData(&module, image.data()));
	hipFunction_t fill, count;
	CHECK(hipModuleGetFunction(&fill, module, "probeDelayedFill"));
	CHECK(hipModuleGetFunction(&count, module, "probeCountFilled"));
	hipStream_t stream, other;
	CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
	CHECK(hipStreamCreateWithFlags(&other, hipStreamNonBlocking));

	const unsigned n = 4096, busyN = 1u << 16;
	void* target; CHECK(hipMalloc(&target, n * sizeof(unsigned)));
	void* scratch; CHECK(hipMalloc(&scratch, busyN * sizeof(unsigned)));
	unsigned* gpuCount; CHECK(hipHostMalloc((void**)&gpuCount, sizeof(unsigned), hipHostMallocMapped));
	std::vector<unsigned> back(n);

	printf("AMD_OPT_FLUSH=%s, %d rounds per row\n", flush ? flush : "(unset, runtime default 1)", rounds);
	struct Row { unsigned pauseUs; bool keepBusy; };
	const Row rows[] = { {0, false}, {200, false}, {500, false}, {1000, false}, {2000, false}, {10000, false}, {10000, true} };
	int totalLost = 0;
	for (const Row& row : rows)
	{
		int lostHost = 0, lostGpu = 0;
		for (int r = 0; r < rounds; ++r)
		{
			CHECK(hipMemsetD32Async(target, 0, n, stream));
			CHECK(hipStreamSynchronize(stream));

			unsigned nArg = n, value = 0x51000000u + unsigned(r) * 0x10000u, spin = 2000;
			void* fillArgs[] = { &target, &nArg, &value, &spin };
			CHECK(hipModuleLaunchKernel(fill, n / 256, 1, 1, 256, 1, 1, 0, stream, fillArgs, nullptr));
			if (row.keepBusy)
			{
				// a kernel on the other stream that outlasts the pause
				unsigned busyArg = busyN, busyValue = 1, busySpin = 20000;
				void* busyArgs[] = { &scratch, &busyArg, &busyValue, &busySpin };
				CHECK(hipModuleLaunchKernel(fill, busyN / 256, 1, 1, 256, 1, 1, 0, other, busyArgs, nullptr));
			}
			if (row.pauseUs) usleep(row.pauseUs);
			CHECK(hipStreamSynchronize(stream));
			CHECK(hipStreamSynchronize(other));

			*gpuCount = 0;
			void* countArgs[] = { &target, &nArg, &value, &gpuCount };
			CHECK(hipModuleLaunchKernel(count, n / 256, 1, 1, 256, 1, 1, 0, stream, countArgs, nullptr));
			CHECK(hipStreamSynchronize(stream));
			CHECK(hipMemcpyDtoH(back.data(), target, n * sizeof(unsigned)));
			unsigned hostCount = 0;
			for (unsigned i = 0; i < n; ++i) if (back[i] == value + i) ++hostCount;
			lostHost += hostCount != n; lostGpu += *gpuCount != n;
		}
		printf("host idle %5u us%s: writes lost in %d rounds (host copy), %d rounds (read by a kernel)\n",
			row.pauseUs, row.keepBusy ? ", GPU kept busy by another stream" : "                                 ", lostHost, lostGpu);
		totalLost += lostHost + lostGpu;
	}
	return totalLost ? 1 : 0;
}
