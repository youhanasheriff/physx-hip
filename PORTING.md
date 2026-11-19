# PhysX GPU on AMD: porting log

Port of the GPU side of NVIDIA PhysX from CUDA to AMD HIP (ROCm), so that PhysX's GPU
pipeline, then ovphysx, then the omni.physx Kit extension can run on AMD GPUs.

This file is the running record: status, decisions, and what has been verified versus
assumed. Every claim is tagged **[ran]** (executed on this machine and observed),
**[read]** (established by reading source or headers) or **[assumed]** (not yet checked).

Last updated: 2026-11-19.

## 1. Status

| Item | State |
|---|---|
| Upstream cloned, base pinned, working branch `hip-port` | done **[ran]** |
| Stock CPU build (clang 21) | done **[ran]** |
| Stock CPU snippets as tests | 55 pass, 0 fail, 4 skipped **[ran]** |
| ROCm 7.1 installed, HIP runs on this PC's integrated GPU | done **[ran]** |
| HIP capability probe (PhysX-style kernels and driver calls) | 135 checks, 0 failed **[ran]** |
| All 61 SDK kernel sources compile and link for AMD GPUs | done **[ran]** |
| All 61 kernel modules load on the GPU, all 488 kernel names resolve | done **[ran]** |
| Host side (context manager, kernel embedding, CMake) built against HIP | not started |
| Any PhysX kernel executed inside a PhysX simulation on AMD | not yet |

Nothing found so far makes the port impossible. See section 9 for what is still unproven.

## 2. Base

- Upstream: `github.com/NVIDIA-Omniverse/PhysX`, remote name `upstream`, push URL disabled.
- **Pinned base: tag `ovphysx-0.8.0`, commit `7d6f6ac8b`, PhysX SDK 5.13.0.** **[ran]**
- Working branch: `hip-port`. Everything is local; nothing has been pushed anywhere.

Why this tag and not `110.3-omni-and-physx-5.9.1` **[ran: git tree comparison]**:

| | `110.3-omni-and-physx-5.9.1` | `ovphysx-0.8.0` |
|---|---|---|
| PhysX SDK | 5.9.1 | 5.13.0 |
| ovphysx source | headers, Python, tests only (no `src/`) | complete, with its test suite |
| Kit extension source (`omni/ovexts`) | present | absent |

Goals 1 (SDK) and 2 (ovphysx) can only be built from the newer tag, and ovphysx's tests are
the only large automated suite in the repo; the SDK ships snippets, not unit tests. Goal 3
(Kit extension) needs the port carried to 5.9.1 or to NVIDIA's next combined release. The
GPU directories differ between the two tags by about 12.9k added / 12.2k removed lines in
118 files, so the port is kept as a compatibility layer with as few edits to NVIDIA's
sources as possible (section 6).

### Licence

The brief said BSD-3. That is only the root `LICENSE.md`. **The PhysX SDK and ovphysx are
Apache-2.0** at both tags: `physx/LICENSE.md`, and all 1,839 SPDX headers under
`physx/source` and `physx/include` say Apache-2.0. **[ran]** Consequences for a fork:

- keep every existing copyright and SPDX line (done; no header has been touched);
- Apache-2.0 section 4(b): modified files must carry a notice that they were changed. Each
  edit to an upstream file is marked with a `Modified for the HIP port` comment, and
  section 10 lists them;
- new files under `hip/` state that they are not part of upstream PhysX. They have no
  licence header yet: **decision needed** on the licence for the port's own files
  (Apache-2.0 is the obvious choice).
- ovphysx's prebuilt binaries are under the NVIDIA Omniverse License; only source is used here.

## 3. Hardware and test strategy

### What this PC can do

Ryzen 9 9950X, integrated Radeon (Granite Ridge, PCI `1002:13c0`), Ubuntu 26.04.1, kernel 7.0.

- Ubuntu 26.04 packages ROCm 7.1 in `universe`. Installed with apt: `hipcc`,
  `libamdhip64-dev`, `rocminfo`, `rocm-smi`, `hipify-clang`, `librocthrust-dev`,
  `rocm-opencl-icd`, `clinfo`. HIP 7.1.52801, compiler is the distro's clang 21.1.8. **[ran]**
- `rocminfo` sees the integrated GPU as **`gfx1036`, wavefront size 32, 2 compute units,
  14.9 GiB addressable** (system RAM through GTT). No `HSA_OVERRIDE_GFX_VERSION` or any
  other override was needed. **[ran]**
- The capability probe (`hip/probe/run.sh`) passes on it: 135 checks, 0 failed. **[ran]**
- Device access: the user was added to the `render` and `video` groups (effective at next
  login). For the current session a temporary ACL was set on `/dev/kfd` and
  `/dev/dri/renderD128`; it disappears at reboot. **[ran]**

So the "no CUDA reference, no supported GPU" problem is smaller than feared: **correctness
work can be done on this PC today.** What it cannot give is performance numbers that mean
anything (2 compute units), coverage of a different GPU generation, or confidence about
memory behaviour on a card with real VRAM.

### Test strategy

1. **CPU pipeline as the numerical reference.** PhysX runs the same scene on its CPU
   pipeline and on its GPU pipeline. There is no NVIDIA GPU here, so the CPU pipeline is the
   only reference. The two pipelines are not bit-identical even on NVIDIA hardware
   (different solvers orderings and float contraction), so comparisons are tolerance-based
   on body poses and velocities after N steps, plus invariants (no NaN, energy bounded,
   contact counts within a band). **[assumed: tolerance levels to be set empirically in M1]**
2. **Snippets as smoke tests.** `hip/scripts/run_snippets.sh` runs every snippet headless.
   The 13 GPU snippets (HelloGRB, PBF, deformables, SDF, isosurface, direct-GPU API) are the
   first GPU targets.
3. **Kernel-level probes.** `hip/probe` checks the primitives PhysX leans on against exact
   host references. It runs in seconds and is the first thing to run on any new GPU.
4. **ovphysx's own test suite** (191 Python tests, 93 C unit tests) once M3 is reached.
5. **HIP-CPU is not part of the plan.** PhysX loads kernels as modules and launches them by
   name through the driver API, and uses warp shuffles and ballots heavily; a CPU
   implementation of HIP would have to emulate all of that. With a working GPU on the desk
   there is no need. *(Maintenance status of HIP-CPU: see research notes, section 11.)*

### Hardware recommendation

*Pending the research task's price data; see section 11.* The shape of it:

- Not needed to start, and not needed for milestone 1.
- Needed before calling anything "works on AMD": one officially supported RDNA discrete
  card, to cover a second GPU generation, real VRAM limits, and realistic performance.
- **Must be an RDNA (Radeon RX) card, not an Instinct card.** See the wavefront constraint
  in section 5. This also rules out most cloud AMD GPU rental, which is Instinct MI300-class.

## 4. How CUDA is used in PhysX (inventory)

Sizes, SDK only **[ran: counted in the tree]**: 61 `.cu` and 59 `.cuh` files, 89,738 lines,
491 `__global__` kernels, of which PhysX looks up 488 by name. Host-side GPU code: 198 files,
79,062 lines. Outside the SDK there are 3 more `.cu` files in ovphysx.

### Kernel compilation and loading **[read]**

- CMake enables the `CUDA` language and compiles each `.cu` with nvcc into an ordinary
  object file, for SASS on sm_70 through sm_120 plus PTX for forward compatibility
  (`source/compiler/cmakegpu/CMakeLists.txt`).
- PhysX does **not** link the CUDA runtime. It defines `__cudaRegisterFatBinary` and
  `__cudaRegisterFunction` itself (`cudamanager/src/CudaKernelWrangler.cpp`). nvcc's
  generated static initialisers call them, which hands PhysX a pointer to each embedded fat
  binary and the name of each kernel (`physxgpu/src/PxgPhysXGpu.cpp`).
- At context creation, `CudaCtxMgr` calls `cuModuleLoadDataEx` on each fat binary.
  `KernelWrangler` then calls `cuModuleGetFunction` for each name in `PxgKernelNames.h`.
- Kernels are launched with `cuLaunchKernel` through the `PxCudaContext` virtual interface
  (495 call sites), with arguments passed as an array of pointers.
- The GPU library is a separate shared object, `libPhysXGpu_64.so`, which the core SDK
  `dlopen`s after `dlopen("libcuda.so.1")` (`physx/src/gpu/PxPhysXGpuModuleLoader.cpp`).

### Driver API surface **[ran: grep counts]**

Almost all driver calls are in one file, `cudamanager/src/CudaContextManager.cpp` (45
distinct `cu*` functions: init, device attributes, context create/push/pop, module load,
streams, events, host/device memory, sync and async copies and memsets, launch). Outside it:
SDF 3D textures in `gpunarrowphase/src/PxgGeometryManager.cpp` (`cuArray3DCreate`,
`cuMemcpy3DAsync`, `cuTexObjectCreate`), an occupancy query, stream priority range, and
OpenGL interop in the snippet renderer.

Host code otherwise uses a small CUDA vocabulary: `CUdeviceptr` (1,159 uses), `CUstream`,
`CUresult`, `CUfunction`, `CUevent`, `CUDA_SUCCESS`, and about 30 `CU_*` constants.

### CUDA language features in kernels **[ran: grep counts over .cu/.cuh]**

| Feature | Uses | HIP status |
|---|---|---|
| `__shfl_sync` / `_xor_` / `_up_` / `_down_` | 644 | HIP's versions demand 64-bit masks; replaced by 32-bit shims **[ran]** |
| `__ballot_sync`, `__any_sync`, `__all_sync` | ~270 | same **[ran]** |
| `WARP_SIZE` (hard-coded 32), `FULL_MASK` (0xffffffff) | 645 | needs 32-lane wavefronts; see section 5 |
| `__syncthreads`, `__shared__` | 290, 535 | native **[ran]** |
| `__syncwarp` | 219 | shim over wave barrier **[ran: compiles; behaviour assumed]** |
| `atomicAdd` (int and float), `atomicMax/Min/Or/And/CAS/Exch` | ~245 | native **[ran]** |
| `__popc`, `__ffs`, `__clz` and 64-bit forms | ~170 | native; one overload added **[ran]** |
| `__launch_bounds__`, `__forceinline__`, `__constant__`, `#pragma unroll` | ~275 | accepted **[ran: compiles]** |
| cache-hinted loads `__ldg/__ldcg/__ldcs` | via wrappers | no AMD equivalent; plain loads **[ran: compiles]** |
| inline PTX assembly | 9 sites, 4 files | rewritten under `__HIPCC__` (section 10) **[ran: compiles]** |
| `tex3D<float>` / texture objects | 2 files (SDF collision) | HIP has texture objects; device reports image support **[ran: attribute only]**, sampling **[assumed]** |
| `float4`, `uint4`, ... vector types | ~5,700 | HIP's types clash with PhysX's operators; CUDA-layout structs supplied **[ran]** |
| `printf`, `assert` in kernels | 583, 273 | compiled out in checked/release; debug builds **[assumed]** |
| Thrust | 1 file, in ovphysx only | rocThrust is installed; not yet tried **[assumed]** |
| `double` in kernels | 99 uses, 10 files | supported by the hardware target **[ran: compiles]** |

Not used at all: cooperative groups, CUB, cuBLAS/cuSPARSE/cuSOLVER, half floats, dynamic
parallelism, unified memory.

## 5. The one hard constraint: 32-lane wavefronts

PhysX's kernels are written for 32-thread warps throughout. AMD GPUs come in two kinds:

- **RDNA (Radeon RX 5000 and later, and Ryzen integrated graphics): 32-lane wavefronts.**
  A wavefront behaves like a CUDA warp, and the kernels port without restructuring.
- **Instinct (MI200/MI300, gfx90a/gfx942) and Vega: 64-lane wavefronts only.** Every warp
  reduction, ballot and scan in PhysX would be wrong there. Supporting them means reworking
  several hundred call sites and is **out of scope**.

`hip/compat/include/PxHipDevice.h` refuses to compile for a 64-lane target. **[ran]**
This is why the hardware recommendation is a Radeon RX card and why MI300 cloud rental does
not help.

## 6. Design of the port

Goal: NVIDIA's sources stay as close to untouched as possible, so the port can follow
upstream releases and move between 5.13 and 5.9.

1. **Stand-in headers** (`hip/compat/include/`). The kernel sources keep including
   `<cuda.h>`, `<vector_types.h>` and so on; under HIP those names resolve to stand-ins that
   map onto HIP. `PxHipDevice.h` holds the device-side layer. **[ran]**
2. **CUDA-layout vector types.** HIP's `float4` family are class templates with their own
   operators, which made 256 operator calls ambiguous against `cutil_math.h`. The stand-ins
   rename HIP's types out of the way while HIP's headers are read, then define plain structs
   with CUDA's sizes and alignments under the usual names, with `static_assert`s on every
   layout. **[ran]** The alignment table (two- and four-component types aligned to their
   size, capped at 16) was written from knowledge of CUDA's header, not copied from it:
   **[assumed]** until checked against a real CUDA toolkit header. PhysX's own struct-size
   asserts pass with it. **[ran]**
3. **Device-only kernel compilation.** Each `.cu` is compiled with
   `clang -x hip --cuda-device-only` straight to a code object. The host never compiles the
   kernel sources, which avoids clang's host/device overloading rules entirely. **[ran]**
4. **Embedding and registration** (not built yet). Each code object is embedded in
   `libPhysXGpu_64.so` as data and registered through the existing
   `PxGpuCudaRegisterFatBinary` table, replacing nvcc's static initialisers. Kernel names
   are resolved by asking each module, instead of relying on `__cudaRegisterFunction`.
5. **Host driver shim** (not built yet). A host-side `cuda.h` stand-in keeps PhysX's own
   opaque handle types (so the public `PxCudaTypes.h` ABI is unchanged) and maps the ~50
   `cu*` calls onto `hip*`. Every one of them has a HIP counterpart in the installed
   headers. **[ran: checked against `hip_runtime_api.h`]**
6. **CMake**: a `PX_GPU_BACKEND` switch (CUDA default, HIP) and a `linux-clang-hip` preset.
7. **HIPIFY is not used on the kernels.** It rewrites sources, which is the opposite of
   goal above. It may be useful once for the host-side shim table.

## 7. Milestones

Sizes are relative; M1 is the unit. "Thread" means one working session like the kickoff.

| # | Milestone | Size | Done when |
|---|---|---|---|
| M0 | Baseline: clone, CPU build and tests, ROCm on this PC, inventory, kernels compile and load | 1 thread | **done** |
| M1 | SDK builds against HIP and runs rigid bodies on GPU | 2-3 threads | `libPhysXGpu_64.so` builds with the HIP backend; `SnippetHelloGRB` runs on the integrated GPU and matches the CPU pipeline within tolerance |
| M2 | Full SDK GPU feature set | 4-6 threads | all 13 GPU snippets pass; a CPU-vs-GPU comparison harness covers rigid bodies, articulations, particles, deformables, SDF; results repeat on a discrete RDNA card |
| M3 | ovphysx on HIP | 4-6 threads | ovphysx builds against the HIP SDK; its C and Python tests pass in GPU mode on AMD, with tensor output on CPU or ROCm devices |
| M4 | omni.physx Kit extension | 3-5 threads, plus risk | the port carried to the SDK version of a released Kit; extension built from `omni/ovexts` source and loaded by Kit |

Main risk per milestone:

- **M1**: first execution of real kernels. Expect wrong results from places where AMD and
  NVIDIA differ silently: reading an inactive lane in a shuffle, barrier semantics, struct
  layout shared between host and device, float contraction.
- **M2**: breadth. 488 kernels; particles and deformables use the densest warp tricks. SDF
  needs 3D texture sampling to work through HIP on RDNA.
- **M3**: ovphysx hands results out as Warp arrays on CUDA devices. Warp is NVIDIA-only, so
  GPU-resident output needs a different path (DLPack with a ROCm device for PyTorch-ROCm,
  or CPU arrays). ovphysx also uses CUDA graphs (stream capture); HIP 7.1 declares the graph
  API **[ran: header check]**, behaviour **[assumed]**. Its driver calls already go through
  a `dlsym` function table (`IOptionalCuda`), which is a clean seam. **[read]**
- **M4**: Kit is closed and ships its own NVIDIA-only plugins. The extension must work with
  an unmodified Kit, on a machine where Kit's renderer cannot start. Whether Kit runs at all
  without an NVIDIA GPU is unknown and may make this milestone unreachable. **[assumed]**

## 8. Decisions needed from Youhana

1. **Test GPU**: whether and when to buy or rent a discrete RDNA card (section 3).
2. **Fork location and visibility**: personal account or the `sheriax` org, public or private.
3. **Licence for the port's own files** under `hip/` (Apache-2.0 suggested).

## 9. Verified versus assumed

Verified by running on this machine:

- stock CPU SDK builds and 55 snippets run clean;
- HIP runs kernels on the Granite Ridge integrated GPU with stock Ubuntu packages;
- module load from memory, lookup by name, argument-array launch, streams, events, async
  copies, pinned host memory;
- warp shuffles (all four kinds), ballots, any/all, warp compaction, shared memory with
  block barriers, int and float atomics, mixed-size kernel arguments: exact match with host
  references;
- every SDK kernel source compiles and links to a code object for gfx1036; all 61 load;
  all 488 kernel names resolve.

Not yet verified:

- that any PhysX kernel computes the right answer (none has been executed yet);
- `__syncwarp`, named-barrier replacement, and the rewritten warp scan under real workloads;
- 3D texture sampling on RDNA through HIP;
- behaviour on any GPU other than gfx1036, and with more than 2 compute units;
- kernel `printf`/`assert` in debug builds;
- CUDA vector alignment table against NVIDIA's header;
- everything about ovphysx and Kit beyond reading.

## 10. Changes to upstream files

All guarded so that the CUDA build is unchanged. Each carries a `Modified for the HIP port`
comment.

| File | Change |
|---|---|
| `physx/source/compiler/cmake/linux/CMakeLists.txt` | `PX_WARNINGS_AS_ERRORS` option (default ON) |
| `physx/snippets/compiler/cmake/linux/SnippetTemplate.cmake`, `SnippetVehicleTemplate.cmake` | `PX_SNIPPETS_HEADLESS` option (default OFF) |
| `physx/include/foundation/PxPreprocessor.h` | `PX_CUDA_COMPILER` also set for `__HIPCC__` |
| `physx/include/PxParticleGpu.h` | no `struct float4;` forward declaration under HIP |
| `physx/source/gpucommon/src/CUDA/atomic.cuh` | `PxRedAddGlobal`: `atomicAdd` instead of PTX `red.global.add.f32` |
| `physx/source/gpunarrowphase/src/CUDA/convexMeshMidphase.cu` | 5 PTX `bar.sync N` barriers become `__syncthreads()` |
| `physx/source/gpunarrowphase/src/CUDA/warpHelpers.cuh` | 2 PTX predicated warp-scan steps rewritten with `__shfl_up_sync` |

## 11. Known issues and notes

- **LLVM 21 AMDGPU backend defect.** `convexCoreCollision.cu` fails at `-O2`/`-O3` on every
  RDNA target tried (gfx1030, gfx1036, gfx1100, gfx1151, gfx1200) with "SOP2/SOPC
  instruction requires too many immediate constants". It builds at `-O1`, which
  `kernel_census.sh` applies to that file only. **[ran]** To do: reduce and report to LLVM,
  and try AMD's own LLVM (`clang-rocm` package).
- Building with clang 21 needs `-DPX_WARNINGS_AS_ERRORS=OFF`; upstream tests clang 14 and 18.
- Four snippets cannot run unattended (need input files or a PVD listener) and are skipped
  by the runner with the reason printed.
- Research on prior art, ROCm support matrix, HIP-CPU status and hardware prices was
  delegated; results to be merged here.

## 12. Reproducing

```sh
# stock CPU build and tests
cd physx && ./generate_projects.sh linux-clang-cpu-only
cd compiler/linux-clang-cpu-only-checked
cmake -DPX_WARNINGS_AS_ERRORS=OFF -DPX_SNIPPETS_HEADLESS=ON . && make -j28
../../../hip/scripts/run_snippets.sh ../../bin/linux.x86_64/checked

# HIP capability probe on the first AMD GPU
hip/probe/run.sh

# compile and link every kernel source for an AMD GPU, then load them all
hip/scripts/kernel_census.sh /tmp/census gfx1036
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -Wno-deprecated-declarations \
    -Iphysx/source/gpucommon/include hip/probe/load_kernels.cpp -o /tmp/load_kernels -lamdhip64
/tmp/load_kernels /tmp/census/*.co
```
