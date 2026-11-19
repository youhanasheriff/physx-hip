// Checks that a kernel on one HIP stream waits for a kernel on another stream when PhysX's
// synchronizeStreams pattern is used: record an event on the first stream, make the second stream wait
// on it, and (as PhysX does) destroy the event straight away.
//
// This file is part of the HIP port and is not part of upstream PhysX.
// build: g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ check_stream_events.cpp -o build/check_stream_events -lamdhip64
// usage: build/check_stream_events build/probe_kernels.<arch>.co

#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <vector>
static std::vector<char> readFile(const char* p){std::vector<char> d; FILE* f=fopen(p,"rb"); fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET); d.resize(s); fread(d.data(),1,s,f); fclose(f); return d;}
int main(int argc,char**argv){
	hipInit(0);
	auto image=readFile(argv[1]); hipModule_t m; hipModuleLoadData(&m,image.data());
	hipFunction_t fAtomics,fShift; hipModuleGetFunction(&fAtomics,m,"probeAtomics"); hipModuleGetFunction(&fShift,m,"probeShift");
	int lo,hi; hipDeviceGetStreamPriorityRange(&lo,&hi);
	hipStream_t A,B; hipStreamCreateWithPriority(&A, hipStreamNonBlocking, hi); hipStreamCreateWithPriority(&B, hipStreamNonBlocking, lo);
	const unsigned N=128;
	void *ds,*dbuf,*dmin,*dbits,*dcas,*dand,*dup,*ddown,*dany; hipMalloc(&ds,4);hipMalloc(&dbuf,N*4);hipMalloc(&dmin,4);hipMalloc(&dbits,4);hipMalloc(&dcas,4);hipMalloc(&dand,4);
	hipMalloc(&dup,N*4);hipMalloc(&ddown,N*4);hipMalloc(&dany,N*4);
	const unsigned big=30000000;
	for(int mode=0;mode<3;++mode){
		int ordered=0; const int runs=10;
		for(int r=0;r<runs;++r){
			hipMemsetD32(dbuf,-1,N);
			unsigned n=big; void* pa[]={&n,&ds,&dbuf,&dmin,&dbits,&dcas,&dand};
			hipModuleLaunchKernel(fAtomics,big/256+1,1,1,256,1,1,0,A,pa,nullptr);   // long kernel on A writes dbuf[0]
			hipEvent_t ev=nullptr;
			if(mode>0){ hipEventCreateWithFlags(&ev, hipEventDisableTiming); hipEventRecord(ev, A); hipStreamWaitEvent(B, ev, 0); if(mode==2) hipEventDestroy(ev); }
			void* pb[]={&dbuf,&dup,&ddown,&dany};
			hipModuleLaunchKernel(fShift,1,1,1,N,1,1,0,B,pb,nullptr);                // kernel on B reads dbuf[0] into dup[1]
			hipStreamSynchronize(B); hipStreamSynchronize(A);
			unsigned up[N]; hipMemcpyDtoH(up,dup,N*4);
			if(up[1]==big-1) ++ordered;
			if(mode==1) hipEventDestroy(ev);
		}
		const char* names[]={"no event (control, expect 0)       ","event kept until both finished      ","event destroyed right after the wait"};
		printf("%s: kernel on stream B ran after stream A's kernel in %d of %d runs\n", names[mode], ordered, runs);
	}
	return 0;
}
