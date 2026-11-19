// Loads PhysX kernel code objects on a GPU and resolves every kernel PhysX asks for by name.
//
// This file is part of the HIP port and is not part of upstream PhysX.
//
// Mirrors what CudaCtxMgr and KernelWrangler do at start-up: load each compiled kernel
// module from memory, then find each name from PxgKernelNames.h in one of the modules.
//
// usage: load_kernels <code-object-file>...
// exit code 0 when every module loads and every kernel name resolves exactly once.

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <string>
#include <vector>

static const char* kernelNames[] = {
#define KERNEL_DEF(id, name) name,
#include "PxgKernelNames.h"
#undef KERNEL_DEF
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

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::printf("usage: %s <code-object-file>...\n", argv[0]);
		return 2;
	}
	if (hipInit(0) != hipSuccess)
	{
		std::printf("hipInit failed\n");
		return 3;
	}

	std::vector<hipModule_t> modules;
	std::vector<std::string> moduleNames;
	std::vector<std::vector<char>> images; // kept alive for as long as the modules are
	int loadFailures = 0;
	for (int i = 1; i < argc; ++i)
	{
		images.push_back(readFile(argv[i]));
		hipModule_t module = nullptr;
		const hipError_t e = images.back().empty() ? hipErrorFileNotFound : hipModuleLoadData(&module, images.back().data());
		if (e != hipSuccess)
		{
			++loadFailures;
			std::printf("LOAD FAIL %s: %d (%s)\n", argv[i], int(e), hipGetErrorString(e));
			continue;
		}
		modules.push_back(module);
		moduleNames.push_back(argv[i]);
	}

	const size_t numKernels = sizeof(kernelNames) / sizeof(kernelNames[0]);
	size_t resolved = 0, missing = 0, duplicated = 0;
	std::vector<unsigned> perModule(modules.size(), 0);
	for (size_t k = 0; k < numKernels; ++k)
	{
		int hits = 0;
		for (size_t m = 0; m < modules.size(); ++m)
		{
			hipFunction_t f = nullptr;
			if (hipModuleGetFunction(&f, modules[m], kernelNames[k]) == hipSuccess && f)
			{
				++hits;
				++perModule[m];
			}
		}
		if (hits == 0)
		{
			++missing;
			std::printf("MISSING   %s\n", kernelNames[k]);
		}
		else
		{
			++resolved;
			if (hits > 1)
			{
				++duplicated;
				std::printf("DUPLICATE %s (in %d modules)\n", kernelNames[k], hits);
			}
		}
	}

	unsigned emptyModules = 0;
	for (size_t m = 0; m < modules.size(); ++m)
		if (perModule[m] == 0)
			++emptyModules;

	for (hipModule_t m : modules)
		(void)hipModuleUnload(m);

	std::printf("---\nmodules: %zu loaded, %d failed, %u provide no named kernel\n", modules.size(), loadFailures, emptyModules);
	std::printf("kernels: %zu named by PhysX, %zu resolved, %zu missing, %zu found in more than one module\n", numKernels, resolved, missing, duplicated);
	return (loadFailures == 0 && missing == 0 && duplicated == 0) ? 0 : 1;
}
