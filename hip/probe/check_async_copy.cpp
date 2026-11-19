// Checks that HIP async host<->device copies follow CUDA's rules for pageable and pinned host memory:
// a pageable source is staged before the call returns, a pageable destination is filled before the call
// returns, and pinned memory is read or written when the copy actually runs.
//
// This file is part of the HIP port and is not part of upstream PhysX.
// build: g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ check_async_copy.cpp -o build/check_async_copy -lamdhip64
// usage: build/check_async_copy build/probe_kernels.<arch>.co

#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstring>
#include <vector>
static std::vector<char> readFile(const char* p){std::vector<char> d; FILE* f=fopen(p,"rb"); fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET); d.resize(s); fread(d.data(),1,s,f); fclose(f); return d;}
int main(int argc,char**argv){
	hipInit(0);
	auto image=readFile(argv[1]); hipModule_t m; hipModuleLoadData(&m,image.data());
	hipFunction_t fAtomics; hipModuleGetFunction(&fAtomics,m,"probeAtomics");
	hipStream_t stream; hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
	const unsigned n=1<<22; void* d; hipMalloc(&d,n*4);
	void *ds,*dmax,*dmin,*dbits,*dcas,*dand; hipMalloc(&ds,4);hipMalloc(&dmax,4);hipMalloc(&dmin,4);hipMalloc(&dbits,4);hipMalloc(&dcas,4);hipMalloc(&dand,4);
	for(int pinned=0;pinned<2;++pinned){
		unsigned* host; std::vector<unsigned> pageable;
		if(pinned) hipHostMalloc((void**)&host,n*4,0); else { pageable.resize(n); host=pageable.data(); }
		for(unsigned i=0;i<n;++i) host[i]=1;
		// keep the stream busy so the copy cannot start straight away
		unsigned big=40000000; void* params[]={&big,&ds,&dmax,&dmin,&dbits,&dcas,&dand};
		hipModuleLaunchKernel(fAtomics,big/256+1,1,1,256,1,1,0,stream,params,nullptr);
		hipMemcpyHtoDAsync(d,host,n*4,stream);
		for(unsigned i=0;i<n;++i) host[i]=2;   // overwrite the source right after the call returns
		hipStreamSynchronize(stream);
		std::vector<unsigned> back(n); hipMemcpyDtoH(back.data(),d,n*4);
		unsigned twos=0; for(unsigned v:back) if(v==2) ++twos;
		printf("%s source: %u of %u elements arrived with the value written AFTER the async copy call returned\n", pinned?"pinned  ":"pageable", twos, n);
		// device -> host: is the data in the destination when the async call returns?
		hipMemsetD32(d, 7, n);
		for(unsigned i=0;i<n;++i) host[i]=0;
		hipModuleLaunchKernel(fAtomics,big/256+1,1,1,256,1,1,0,stream,params,nullptr);
		hipMemcpyDtoHAsync(host,d,n*4,stream);
		unsigned arrived=0; for(unsigned i=0;i<n;++i) if(host[i]==7) ++arrived;
		printf("%s destination: %u of %u elements present when the async device-to-host call returned\n", pinned?"pinned  ":"pageable", arrived, n);
		hipStreamSynchronize(stream);
	}
	return 0;
}
