# PhysX GPU on AMD: porting log

Port of the GPU side of NVIDIA PhysX from CUDA to AMD HIP (ROCm), so that PhysX's GPU
pipeline, then ovphysx, then the omni.physx Kit extension can run on AMD GPUs.

This file is the running record: status, decisions, and what has been verified versus
assumed. Claims are tagged **[ran]** (executed on this machine and observed), **[read]**
(established by reading source or headers), **[research]** (from the delegated web research
of 2026-11-19, with its sources; not re-checked here) or **[assumed]** (not yet checked).

Last updated: 2026-11-19, end of the kickoff thread.

## 1. Status

| Item | State |
|---|---|
| Upstream cloned, base pinned, working branch `hip-port` | done **[ran]** |
| Stock CPU build (clang 21) and snippets | 55 pass, 0 fail, 4 skipped **[ran]** |
| ROCm 7.1 installed; HIP runs on this PC's integrated GPU (gfx1036) | done **[ran]** |
| HIP capability probe (PhysX-style kernels and driver calls) | 135 checks, 0 failed **[ran]** |
| All 61 SDK kernel sources compile and link for AMD GPUs | done **[ran]** |
| `libPhysXGpu_64.so` builds with the HIP backend through CMake | done **[ran]** |
| Rigid bodies on the GPU pipeline match the CPU pipeline | 8 of 8 scenes pass, PGS and TGS **[ran]** |
| All snippets against the HIP build | 67 pass, 2 fail, 4 skipped **[ran]** |
| GPU snippets (14) | 12 run clean, 2 fail intermittently (section 8) **[ran]** |
| Numerical validation of GPU-only features (particles, deformables, SDF) | not started |
| ovphysx, Kit extension | not started |

**Verdict so far: feasible.** Nothing found makes the port impossible. PhysX's GPU rigid-body
pipeline runs on an AMD GPU today and agrees with the CPU pipeline. One open defect blocks
calling milestone 1 complete: an ordering race that crashes two GPU snippets (section 8).

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
118 files, so the port is kept as a compatibility layer with few edits to NVIDIA's sources.

### Licence

The brief said BSD-3. That is only the root `LICENSE.md`. **The PhysX SDK and ovphysx are
Apache-2.0** at both tags: `physx/LICENSE.md`, and all 1,839 SPDX headers under
`physx/source` and `physx/include`. **[ran]** Consequences for a fork:

- keep every existing copyright and SPDX line (done; no header has been touched);
- Apache-2.0 section 4(b): modified files must say they were changed. Every edit to an
  upstream file carries a `Modified for the HIP port` comment; section 10 lists them;
- new files under `hip/` say they are not part of upstream PhysX. They have no licence
  header yet: **decision needed** (Apache-2.0 is the obvious choice);
- ovphysx's prebuilt binaries are under the NVIDIA Omniverse License; only source is used.

## 3. Hardware and test strategy

### What this PC can do

Ryzen 9 9950X, integrated Radeon (Granite Ridge, PCI `1002:13c0`), Ubuntu 26.04.1, kernel 7.0.

- Ubuntu 26.04 packages ROCm 7.1 in `universe`. Installed with apt: `hipcc`,
  `libamdhip64-dev`, `rocminfo`, `rocm-smi`, `hipify-clang`, `librocthrust-dev`,
  `rocm-opencl-icd`, `clinfo`. HIP 7.1.52801; the compiler is the distro's clang 21.1.8. **[ran]**
- The integrated GPU is **`gfx1036`, wavefront size 32, 2 compute units, 14.9 GiB
  addressable** (system RAM). No `HSA_OVERRIDE_GFX_VERSION` or other override needed. **[ran]**
- It is not in AMD's official support matrix, but AMD's own build system marks gfx1036 as
  sanity tested and release ready, and AMD publishes device packages for it. **[research]**
- Device access: the user was added to the `render` and `video` groups (effective at next
  login). For the current boot a temporary ACL was set on `/dev/kfd` and
  `/dev/dri/renderD128`; it disappears at reboot. **[ran]**
- The newest ROCm is 10.2.0 (2026-11-18), which officially supports Ubuntu 26.04.1 and can
  be installed without root from pip or tarballs. **[research]** Not tried; 7.1 is what runs here.

**Correctness work can be done on this PC.** What it cannot give: performance numbers that
mean anything (PBF takes 279 s here), coverage of a second GPU generation, behaviour on a
card with real VRAM, and behaviour with more than 2 compute units, where races that hide
here may show.

### Test strategy

1. **CPU pipeline as the numerical reference for rigid bodies and articulations.**
   `hip/tests/cpu_gpu_compare` runs the same scene on both pipelines and compares poses.
   The two are not bit-identical even on NVIDIA hardware, so each scene has a tolerance
   that fits it. The sharpest check is the *converged* variant: at 64 solver iterations
   both solvers reach the same resting state, so a 6 cm difference in a 12-level pyramid at
   the default 4 iterations shrinks to 1.1 mm (PGS) and 0.1 mm (TGS). That shows the
   default-iteration gap is solver convergence, not a defect. **[ran]**
2. **GPU-only features have no CPU reference.** PBD particles, deformable surfaces and
   volumes, and SDF collision exist only on the GPU. For these the plan is invariants
   (finite state, volume and momentum conservation, analytic rest states) and comparison
   between two different AMD GPUs. "Snippet ran without errors" is all that is established
   for them today. **[assumed plan]**
3. **Snippets as smoke tests**: `hip/scripts/run_snippets.sh`.
4. **Runtime-semantics probes** (`hip/probe`): exact checks of the primitives PhysX leans
   on, plus checks of HIP behaviours PhysX depends on implicitly. Seconds to run; the first
   thing to run on any new GPU.
5. **ovphysx's own tests** (191 Python, 93 C) at milestone 3.
6. **HIP-CPU is out.** It was archived on 2026-07-07, has no `hipModuleLoadData`, and its
   warp size is the CPU's SIMD width. **[research]** chipStar on a CPU lacks inline-assembly
   handling and 3D textures. Neither is needed with a working GPU on the desk.

### Hardware recommendation

Not needed now, and not needed to finish milestone 1. Needed before anyone says "works on
AMD": one officially supported RDNA card, for a second GPU generation, real VRAM limits,
more compute units and usable performance. Prices **[research, Nov 2026, volatile]**:

| Option | Cost | Notes |
|---|---|---|
| Rent Radeon AI PRO R9700 (gfx1201) at HOSTKEY | about EUR 0.47/hour | cheapest route to an officially supported RDNA4 GPU; whether sign-up is self-serve is unverified |
| Buy RX 7600 8 GB (gfx1102) | about USD 300-390 | cheapest officially supported card; 8 GB limits scene size |
| Buy RX 9060 XT 16 GB (gfx1200) | about USD 730 street, INR 60-80k in India | newest generation, 16 GB; prices rising |
| Rent Instinct MI300X (DigitalOcean, Hot Aisle) | USD 2-3/hour | **no use here**: 64-lane wavefronts, see section 5 |

Recommendation: when milestone 2 starts, rent the R9700 for a few hours first (a few euros)
to validate on a second, officially supported GPU. Buy a card only if regular performance
work is wanted; then the RX 9060 XT 16 GB is the better fit, the RX 7600 the cheaper one.
Any of them fits the 9950X board; the integrated GPU can stay as a second test target.

## 4. How CUDA is used in PhysX (inventory)

Sizes, SDK only **[ran]**: 61 `.cu` and 59 `.cuh` files, 89,738 lines, 491 `__global__`
kernels, of which PhysX looks up 488 by name. Host-side GPU code: 198 files, 79,062 lines.
Outside the SDK there are 3 more `.cu` files in ovphysx.

### Kernel compilation and loading **[read]**

- CMake enables the `CUDA` language and compiles each `.cu` with nvcc for sm_70..sm_120.
- PhysX does not link the CUDA runtime. It defines `__cudaRegisterFatBinary` and
  `__cudaRegisterFunction` itself; nvcc's generated static initialisers call them, handing
  PhysX each embedded fat binary and kernel name.
- `CudaCtxMgr` calls `cuModuleLoadDataEx` per fat binary; `KernelWrangler` calls
  `cuModuleGetFunction` per name in `PxgKernelNames.h`.
- Kernels are launched with `cuLaunchKernel` through the `PxCudaContext` virtual interface
  (495 call sites), arguments passed as an array of pointers.
- Each `.cu` also defines an empty host function (`initSolverKernels3()` and so on, 61 in
  all) that host code calls to force the object file to link.
- The GPU library is a separate `libPhysXGpu_64.so`, `dlopen`ed after `libcuda.so.1`.

### Driver API surface **[ran]**

45 distinct `cu*` functions, nearly all in `cudamanager/src/CudaContextManager.cpp`. Outside
it: SDF 3D textures in `PxgGeometryManager.cpp`, an occupancy query, stream priorities, and
OpenGL interop in the snippet renderer. Every call has a HIP counterpart except
`cuMemcpyPeerAsync`, which is mapped to a device-to-device copy. All 15 streams PhysX
creates are non-blocking.

### CUDA language features in kernels **[ran: grep counts]**

| Feature | Uses | HIP status |
|---|---|---|
| `__shfl_sync` and `_xor_`/`_up_`/`_down_` forms | 644 | HIP's demand 64-bit masks; 32-bit shims supplied **[ran]** |
| `__ballot_sync`, `__any_sync`, `__all_sync` | ~270 | same **[ran]** |
| `WARP_SIZE` 32, `FULL_MASK` 0xffffffff | 645 | needs 32-lane wavefronts; section 5 |
| `__syncthreads`, `__shared__` | 290, 535 | native **[ran]** |
| `__syncwarp` | 219 | shim over wave barrier **[ran in simulation; not isolated]** |
| `atomicAdd` (int, float), `atomicMax/Min/Or/And/CAS/Exch` | ~245 | native **[ran]** |
| `__popc`, `__ffs`, `__clz`, 64-bit forms | ~170 | native; one overload added **[ran]** |
| `__launch_bounds__`, `__forceinline__`, `__constant__`, `#pragma unroll` | ~275 | accepted **[ran]** |
| cache-hinted loads `__ldg/__ldcg/__ldcs` | via wrappers | plain loads **[ran]** |
| inline PTX assembly | 9 sites, 4 files | rewritten under `__HIPCC__`, section 10 **[ran]** |
| `tex3D<float>`, texture objects | 2 files (SDF) | SDF snippet runs clean; values not checked **[ran]** |
| `float4`, `uint4`, ... | ~5,700 | CUDA-layout structs supplied **[ran]** |
| device function pointers | convexCore, particles | compile; these broke 12 kernels in the ZLUDA attempt **[research]**; not isolated here |
| `printf`, `assert` in kernels | 583, 273 | compiled out in checked/release; debug **[assumed]** |
| Thrust | 1 file, ovphysx only | rocThrust installed; untried **[assumed]** |

Not used: cooperative groups, CUB, cuBLAS/cuSPARSE/cuSOLVER, half floats, unified memory.

## 5. Constraints and findings that shape the port

### 32-lane wavefronts only

PhysX's kernels assume 32-thread warps everywhere. RDNA GPUs (Radeon RX 5000 and later,
Ryzen integrated graphics) run 32-lane wavefronts and behave like CUDA warps. Instinct
(MI200/MI300) and Vega run 64-lane wavefronts only; every warp reduction, ballot and scan
would be wrong there, and supporting them means reworking several hundred call sites.
**Out of scope.** The build refuses gfx8/gfx9 targets. **[ran]**

### NVIDIA behaviours PhysX relies on without saying so

Each of these compiled cleanly and failed only at run time.

1. **Oversized shifts.** PTX clamps shift amounts, so `1u << 40` is 0 on NVIDIA. AMD (and
   x86) use the low bits. The radix sort builds counters with `1u << ((radix - i) << 3)`
   and needs the out-of-range terms to vanish; without that the broad phase faulted on its
   first sort. Fix: `hip/ptx-semantics`, an LLVM pass loaded into every kernel compile that
   rewrites variable shifts to PTX behaviour. No source changes. **[ran]**
2. **No type-based alias analysis.** The kernels cast between `float4`, `PxVec3` and raw
   bytes; nvcc tolerates it, clang at `-O3` miscompiled it (garbage contact-manager
   pointers out of the narrow phase). Fix: `-fno-strict-aliasing` for kernels. **[ran]**
3. **Finite maths.** nvcc's `-use_fast_math` does not assume finite values; clang's
   `-ffast-math` does. Kernels are built with `-ffp-contract=fast` only. **[read]**

More of this class should be expected in the parts not yet validated numerically.
Candidates not yet examined: float-to-int conversion out of range (PTX saturates), integer
division by zero, reads of lanes that are inactive in a shuffle.

### HIP behaviours checked and found to match CUDA **[ran]**

- async copies with pageable and pinned host memory (`hip/probe/check_async_copy.cpp`);
- a stream waiting on another stream's kernel through an event, including PhysX's habit of
  destroying the event right after requesting the wait (`check_stream_events.cpp`).

## 6. Design of the port

Goal: NVIDIA's sources stay close to untouched, so the port can follow upstream releases.

1. **Stand-in headers** (`hip/compat/include/`). Sources keep including `<cuda.h>`,
   `<vector_types.h>` and so on; under HIP those resolve to stand-ins. `PxHipDevice.h` is
   the kernel side, `PxHipHost.h` the host side. **[ran]**
2. **CUDA-layout vector types.** HIP's `float4` family are class templates whose operators
   clash with `cutil_math.h`. The stand-ins rename HIP's types out of the way while HIP's
   headers are read, then define plain structs with CUDA's layout, each with a
   `static_assert`. The alignment table was written from knowledge of CUDA's header, not
   copied: **[assumed]** until compared with a real CUDA toolkit header. PhysX's own
   struct-size asserts pass with it on host and device. **[ran]**
3. **Device-only kernel compilation.** Each `.cu` is compiled with
   `clang -x hip --cuda-device-only` to a code object. The host never compiles kernel
   sources, which sidesteps clang's host/device overloading rules. **[ran]**
4. **Embedding and registration.** `hip/cmake/PxHipBackend.cmake` embeds each code object
   in the GPU library with `.incbin`, registers it through the existing
   `PxGpuCudaRegisterFatBinary` table, and defines the 61 empty host functions. Kernel
   names are resolved by asking each module. **[ran]**
5. **Host driver shim.** `PxHipHost.h` keeps PhysX's own opaque handle types, so the
   public `PxCudaTypes.h` ABI is unchanged, and maps the `cu*` calls onto `hip*`. **[ran]**
6. **CMake.** `PX_GPU_BACKEND` (CUDA default, HIP) and a `linux-clang-hip` preset.
   `PX_HIP_ARCHITECTURES` picks GPU targets; default is whatever `rocminfo` finds. **[ran]**
7. **HIPIFY is not used.** It rewrites sources, the opposite of the goal.

The port so far: 6 commits on `hip-port`, 16 upstream files modified and 1 added (section
10), everything else new under `hip/`.

## 7. Milestones

Sizes are relative. "Thread" means one working session like the kickoff.

| # | Milestone | Size | Done when |
|---|---|---|---|
| M0 | Baseline, inventory, kernels compile and load | 1 thread | **done** |
| M1 | SDK builds against HIP and runs rigid bodies on GPU | 2-3 threads | **mostly done in the kickoff thread.** Remaining: fix the ordering race (section 8); articulation scenes in the comparison test; multi-architecture build |
| M2 | Full SDK GPU feature set | 4-6 threads | all 14 GPU snippets pass repeatedly; invariant-based tests for particles, deformables and SDF; same results on a discrete RDNA card |
| M3 | ovphysx on HIP | 4-6 threads | ovphysx builds against the HIP SDK; its C and Python tests pass in GPU mode on AMD, with tensor output on CPU or ROCm devices |
| M4 | omni.physx Kit extension | 3-5 threads, plus risk | port carried to the SDK version of a released Kit; extension built from `omni/ovexts` and loaded by Kit |

Main risk per milestone:

- **M1**: the open race. Until its cause is known it is not clear whether it is in the
  shim, in HIP, or an ordering PhysX gets for free on NVIDIA.
- **M2**: breadth, and no CPU reference for the GPU-only features. More silent
  NVIDIA-only behaviours are likely. Function-pointer kernels need isolating.
- **M3**: ovphysx hands results out as Warp arrays on CUDA devices. Warp is NVIDIA-only
  (AMD's ROCm port of Warp covers Instinct only **[research]**), so GPU-resident output
  needs another path: DLPack with a ROCm device for PyTorch-ROCm, or CPU arrays. ovphysx
  also uses CUDA graphs; HIP 7.1 declares the graph API **[ran: header]**, behaviour
  **[assumed]**. Its driver calls already go through a `dlsym` function table
  (`IOptionalCuda`), a clean seam. **[read]**
- **M4**: Kit is closed and ships NVIDIA-only plugins. The extension must work with an
  unmodified Kit on a machine where Kit's renderer cannot start. Whether Kit runs at all
  without an NVIDIA GPU is unknown and may make this milestone unreachable. **[assumed]**

## 8. Open defects

### Ordering race in two GPU snippets (blocks M1)

`SnippetDeformableVolumeKinematic` and `SnippetDirectGPUAPIArticulation` fail
intermittently in normal asynchronous mode. **[ran]**

- `SnippetDeformableVolumeKinematic`: 10 of 10 runs crash. The host dereferences a garbage
  contact-manager pointer from the narrow phase's lost/found list
  (`PxgConstraintPartition.cpp:1387`).
- `SnippetDirectGPUAPIArticulation`: a GPU memory fault (`HSA_STATUS_ERROR_MEMORY_FAULT`),
  then a hang. Intermittent; it also passed on several runs.
- Both pass 10 of 10 with `PHYSX_GPU_LAUNCH_SYNCHRONOUS=1` (PhysX waits on each launch) and
  with `AMD_SERIALIZE_KERNEL=3` (HIP waits on each kernel). `AMD_SERIALIZE_COPY=3` does not
  help; `HIP_HOST_COHERENT=1` and `AMD_DIRECT_DISPATCH=0` do not fix it.
- So something proceeds before a kernel has finished. Ruled out by direct test: pageable
  and pinned async copy rules, and event-based stream ordering with early event destruction.
- Not yet examined: reads of host-mapped pinned memory that kernels write directly
  (`*Mapped` buffers), `cuStreamQuery` used as a flush, context push/pop across worker
  threads (HIP auto-creates and pushes a primary context, unlike CUDA **[research]**),
  re-recording one event object every frame, and launches PhysX expects to fail with
  `CUDA_ERROR_INVALID_VALUE` for empty grids.
- The rigid-body comparison test and the other 12 GPU snippets pass in asynchronous mode.
  With only 2 compute units here, a race can hide; treat "passes on gfx1036" with care.

Next step: trace HIP API calls (`AMD_LOG_LEVEL=3`) for one failing frame of
`SnippetDeformableVolumeKinematic` and find the read that precedes its kernel.

### Smaller items

- **LLVM 21 AMDGPU backend defect.** `convexCoreCollision.cu` fails at `-O2`/`-O3` on every
  RDNA target tried with "SOP2/SOPC instruction requires too many immediate constants".
  Built at `-O1` for that file only. To do: reduce and report; try AMD's LLVM (`clang-rocm`).
- `SnippetPBF` takes 279 s on this GPU (it passes). Not investigated; may just be 2 compute units.
- The snippet renderer's CUDA/OpenGL buffer sharing is disabled under HIP.
- Only gfx1036 code objects are built by default; multi-target builds are untested.
- Windows is untouched.

## 9. Decisions needed from Youhana

1. **Test GPU**: rent or buy, and when (section 3).
2. **Fork location and visibility**: personal account or the `sheriax` org, public or private.
3. **Licence for the port's own files** under `hip/` (Apache-2.0 suggested).
4. **sudo for future sessions**: the 1Password item linked in the kickoff thread is in a
   vault the service-account token cannot read. Move or copy it to a readable vault, or
   keep answering prompts.

## 10. Changes to upstream files

All guarded so that the CUDA build is unchanged; each carries a `Modified for the HIP port`
comment. The CUDA build itself has not been re-run (no CUDA toolkit here). **[assumed intact]**

| File | Change |
|---|---|
| `physx/compiler/public/CMakeLists.txt` | `PX_GPU_BACKEND`; HIP backend instead of the CUDA language |
| `physx/source/compiler/cmakegpu/CMakeLists.txt` | no `CUDA` project language under HIP |
| `physx/source/compiler/cmakegpu/linux/CMakeLists.txt` | kernels of each GPU library built through the HIP backend |
| `physx/source/compiler/cmake/linux/CMakeLists.txt` | `PX_WARNINGS_AS_ERRORS` option; `PX_GPU_BACKEND_HIP` define |
| `physx/snippets/compiler/cmake/linux/SnippetTemplate.cmake`, `SnippetVehicleTemplate.cmake` | `PX_SNIPPETS_HEADLESS` option |
| `physx/snippets/compiler/cmake/linux/SnippetRender.cmake` | HIP runtime instead of the CUDA driver library |
| `physx/snippets/snippetrender/SnippetRender.cpp` | no CUDA/OpenGL interop under HIP |
| `physx/buildtools/presets/public/linux-clang-hip.xml` | new preset |
| `physx/include/foundation/PxPreprocessor.h` | `PX_CUDA_COMPILER` for `__HIPCC__`; `PX_GPU_BACKEND_HIP` default |
| `physx/include/PxParticleGpu.h` | no `struct float4;` forward declaration under HIP |
| `physx/source/cudamanager/src/CudaContextManager.cpp` | HIP runtime version check; `PHYSX_GPU_LAUNCH_SYNCHRONOUS` |
| `physx/source/cudamanager/src/CudaKernelWrangler.cpp` | kernel lookup per module; nvcc registration hooks excluded |
| `physx/source/physx/src/gpu/PxPhysXGpuModuleLoader.cpp` | no `libcuda.so.1` preload under HIP |
| `physx/source/gpucommon/src/CUDA/atomic.cuh` | `PxRedAddGlobal`: `atomicAdd` instead of PTX `red.global.add.f32` |
| `physx/source/gpunarrowphase/src/CUDA/convexMeshMidphase.cu` | 5 PTX `bar.sync N` barriers become `__syncthreads()` |
| `physx/source/gpunarrowphase/src/CUDA/warpHelpers.cuh` | 2 PTX predicated warp-scan steps rewritten with `__shfl_up_sync` |

## 11. Prior art **[research]**

No native HIP, SYCL, OpenCL or Vulkan port of PhysX 5 GPU was found. Two related efforts:

- `github.com/eugenesflanagin-spec/physx5-zluda` (Oct 2026, one author): PhysX 5.11 built
  PTX-only and run through ZLUDA on an RX 9070 XT with an `LD_PRELOAD` shim. 12 of 488
  kernels fail (function pointers, in convexCore and particles). Validation is
  "trajectories agree qualitatively".
- SCALE (Spectral Compute) lists PhysX in its validation suite as tested and not passing;
  its build stops at the PTX `bar` instructions. gfx1036 is not a SCALE target.

ZLUDA's own PhysX work targets 32-bit legacy game PhysX. Isaac Sim skips non-NVIDIA GPUs.
The search was one pass on 2026-11-19; Reddit and Hacker News were not searched in depth.

## 12. Reproducing

```sh
# stock CPU build and tests
cd physx && ./generate_projects.sh linux-clang-cpu-only
cd compiler/linux-clang-cpu-only-checked
cmake -DPX_WARNINGS_AS_ERRORS=OFF -DPX_SNIPPETS_HEADLESS=ON . && make -j28
../../../hip/scripts/run_snippets.sh ../../bin/linux.x86_64/checked

# HIP build (overwrites the same bin directory)
cd physx && ./generate_projects.sh linux-clang-hip
make -C compiler/linux-clang-hip-checked -j28

# tests against the HIP build
hip/probe/run.sh                                        # runtime primitives
hip/tests/build.sh && physx/bin/linux.x86_64/checked/cpu_gpu_compare [--tgs]
hip/scripts/run_snippets.sh physx/bin/linux.x86_64/checked

# debugging aids
PHYSX_GPU_LAUNCH_SYNCHRONOUS=1 <program>   # PhysX waits on every kernel launch
AMD_SERIALIZE_KERNEL=3 <program>            # HIP waits on every kernel
AMD_LOG_LEVEL=3 <program>                   # HIP API trace
```

Note: the CPU-only and HIP presets write to the same `physx/bin/linux.x86_64/<config>`
directory, so building one replaces the other's binaries.
