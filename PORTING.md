# PhysX GPU on AMD: porting log

Port of the GPU side of NVIDIA PhysX from CUDA to AMD HIP (ROCm), so that PhysX's GPU
pipeline, then ovphysx, then the omni.physx Kit extension can run on AMD GPUs.

This file is the running record: status, decisions, and what has been verified versus
assumed. Claims are tagged **[ran]** (executed on the test machine and observed), **[read]**
(established by reading source or headers), **[research]** (from a web search on
2026-11-19; not re-checked here) or **[assumed]** (not yet checked).

The port is unofficial and is not affiliated with NVIDIA or AMD.

Last updated: 2026-11-20.

## 1. Status

| Item | State |
|---|---|
| Upstream cloned, base pinned, working branch `hip-port` | done **[ran]** |
| Stock CPU build (clang 21) and snippets | 55 pass, 0 fail, 4 skipped **[ran]** |
| ROCm 7.1 installed; HIP runs on the test machine's integrated GPU (gfx1036) | done **[ran]** |
| HIP capability probe (PhysX-style kernels and driver calls) | 135 checks, 0 failed **[ran]** |
| All 61 SDK kernel sources compile and link for AMD GPUs | done **[ran]** |
| `libPhysXGpu_64.so` builds with the HIP backend through CMake | done **[ran]** |
| Rigid bodies on the GPU pipeline match the CPU pipeline | 8 of 8 scenes pass, PGS and TGS **[ran]** |
| All snippets against the HIP build | 69 pass, 0 fail, 4 skipped **[ran]** |
| GPU snippets (14) | all pass, with two runtime workarounds in place (section 8) **[ran]** |
| Numerical validation of GPU-only features (particles, deformables, SDF) | not started |
| ovphysx, Kit extension | not started |

**Verdict so far: feasible.** Nothing found makes the port impossible. PhysX's GPU rigid-body
pipeline runs on an AMD GPU today and agrees with the CPU pipeline, and every GPU snippet
runs to completion. Two defects below PhysX were found on the way, both in how the HIP
runtime and the driver behave on the test GPU, and both are worked around in the port
(section 8). All of this is on one integrated GPU; nothing has run on a discrete card yet.

## 2. Base

- Upstream: `github.com/NVIDIA-Omniverse/PhysX`.
- **Pinned base: tag `ovphysx-0.8.0`, commit `7d6f6ac8b`, PhysX SDK 5.13.0.** **[ran]**
- Working branch: `hip-port`.
- This repository: `github.com/youhanasheriff/physx-hip`, public since 2026-11-19. It is a
  standalone repository, not a GitHub fork of NVIDIA's.

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

The root `LICENSE.md` is BSD-3-Clause, but **the PhysX SDK and ovphysx are Apache-2.0** at
both tags: `physx/LICENSE.md`, and all 1,839 SPDX headers under `physx/source` and
`physx/include`. **[ran]** Consequences for a fork:

- keep every existing copyright and SPDX line (done; no header has been touched);
- Apache-2.0 section 4(b): modified files must say they were changed. Every edit to an
  upstream file carries a `Modified for the HIP port` comment; section 10 lists them;
- new files under `hip/` say they are not part of upstream PhysX and are Apache-2.0, with
  their own copyright line and SPDX header; the licence text is in `hip/LICENSE`;
- ovphysx's prebuilt binaries are under the NVIDIA Omniverse License; only source is used.

## 3. Hardware and test strategy

### The test machine

Ryzen 9 9950X, integrated Radeon (Granite Ridge, PCI `1002:13c0`), Ubuntu 26.04.1, kernel 7.0.
No discrete GPU.

- Ubuntu 26.04 packages ROCm 7.1 in `universe`. Installed with apt: `hipcc`,
  `libamdhip64-dev`, `rocminfo`, `rocm-smi`, `hipify-clang`, `librocthrust-dev`,
  `rocm-opencl-icd`, `clinfo`. HIP 7.1.52801; the compiler is the distro's clang 21.1.8. **[ran]**
- The integrated GPU is **`gfx1036`, wavefront size 32, 2 compute units, 14.9 GiB
  addressable** (system RAM). No `HSA_OVERRIDE_GFX_VERSION` or other override needed. **[ran]**
- It is not in AMD's official support matrix, but AMD's own build system marks gfx1036 as
  sanity tested and release ready, and AMD publishes device packages for it. **[research]**
- Device access: the user must be in the `render` and `video` groups to open `/dev/kfd`
  and the render node. **[ran]**
- The newest ROCm is 10.2.0 (2026-11-18), which officially supports Ubuntu 26.04.1 and can
  be installed without root from pip or tarballs. **[research]** Not tried; 7.1 is what was used.

**Correctness work can be done on this machine.** What it cannot give: performance numbers that
mean anything, coverage of a second GPU generation, behaviour on a
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
   handling and 3D textures. Neither is needed with a working GPU at hand.

### Hardware still needed

Nothing more is needed to finish milestone 1. Before anyone can say "works on AMD", the
port has to run on at least one officially supported RDNA card: a second GPU generation,
real VRAM limits, more compute units and usable performance. Options
**[research, Nov 2026, prices volatile]**:

| Option | Cost | Notes |
|---|---|---|
| Rent Radeon AI PRO R9700 (gfx1201) at HOSTKEY | about EUR 0.47/hour | cheapest route to an officially supported RDNA4 GPU; whether sign-up is self-serve is unverified |
| Buy RX 7600 8 GB (gfx1102) | about USD 300-390 | cheapest officially supported card; 8 GB limits scene size |
| Buy RX 9060 XT 16 GB (gfx1200) | about USD 730 street, INR 60-80k in India | newest generation, 16 GB; prices rising |
| Rent Instinct MI300X (DigitalOcean, Hot Aisle) | USD 2-3/hour | **no use here**: 64-lane wavefronts, see section 5 |

Plan: validate on a second, officially supported GPU when milestone 2 starts. Which one is
not decided. **Reports from anyone who builds this on a discrete Radeon card are welcome**:
the output of `hip/probe/run.sh`, `cpu_gpu_compare` and `hip/scripts/run_snippets.sh` is
what is needed.

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

### HIP behaviour that differs from CUDA

- **A kernel's results are not in memory until the host waits.** By default the HIP runtime
  leaves what a kernel wrote in the GPU's caches and flushes them only when the host waits
  on the device. On the test GPU those results are lost if the GPU goes idle first. The
  GPU library switches the runtime to flush after every kernel. Section 8 has the details.
  **[ran]** (`hip/probe/check_idle_gpu_keeps_writes.cpp`)
- **Streams cost hardware queues, and a small GPU has few.** A CUDA stream is cheap. The HIP
  runtime backs streams with hardware queues, up to 4 for each of three priorities plus one
  of its own, and PhysX uses all three priorities. The test GPU offers a process 8 compute
  queues; a scene that reached 8 stopped for good. The GPU library caps the runtime's
  queues to stay below what the GPU offers. Section 8 has the details. **[ran]**

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

The port so far: 17 upstream files modified and 1 added (section 10), everything else new
under `hip/`.

## 7. Milestones

Sizes are relative: one unit is the work milestone 0 took.

| # | Milestone | Size | Done when |
|---|---|---|---|
| M0 | Baseline, inventory, kernels compile and load | 1 | **done** |
| M1 | SDK builds against HIP and runs rigid bodies on GPU | 2-3 | **mostly done.** Remaining: articulation scenes in the comparison test; multi-architecture build |
| M2 | Full SDK GPU feature set | 4-6 | all 14 GPU snippets pass repeatedly; invariant-based tests for particles, deformables and SDF; same results on a discrete RDNA card |
| M3 | ovphysx on HIP | 4-6 | ovphysx builds against the HIP SDK; its C and Python tests pass in GPU mode on AMD, with tensor output on CPU or ROCm devices |
| M4 | omni.physx Kit extension | 3-5, plus risk | port carried to the SDK version of a released Kit; extension built from `omni/ovexts` and loaded by Kit |

Main risk per milestone:

- **M1**: the crash that blocked it is explained and worked around (section 8). What is
  left is test coverage, not a known defect.
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

## 8. Defects

### Kernel results lost when the GPU goes idle (found, worked around)

`SnippetDeformableVolumeKinematic` and `SnippetDirectGPUAPIArticulation` used to fail on
most runs. The first crashed on the host, on a null contact-manager pointer read back from
the narrow phase; the second took a GPU memory fault at address 0. It looked like an
ordering race. It is not one. **[ran]** unless marked:

- By default the HIP runtime dispatches kernels with device-scope fences
  (`AMD_OPT_FLUSH=1`). What a kernel writes stays in the GPU's caches, and the runtime
  flushes them to memory only when the host waits on the device, with a system-scope
  barrier packet. **[read]** (`rocclr/device/rocm/rocvirtual.cpp`, ROCm 7.1)
- On gfx1036, results left in the caches are lost if the GPU then sits idle for about a
  millisecond. `hip/probe/check_idle_gpu_keeps_writes.cpp` shows it with one stream and one
  kernel: launch the kernel, leave the host idle, then wait and read back.

  | Host idle before the wait | Rounds that lost the kernel's writes (of 20) |
  |---|---|
  | 0, 0.2, 0.5 ms | 0 |
  | 1 ms | 19 |
  | 2 ms, 10 ms | 20 |
  | 10 ms, a second stream keeps the GPU busy | 0 |
  | any of the above with `AMD_OPT_FLUSH=0` | 0 |

  A second kernel reading the buffer sees the same loss as a copy to the host, so the data
  is gone, not just invisible to the host.
- PhysX runs into it because it launches kernels on one stream and then does a millisecond
  or more of host work before it waits for them. In the failing frame that work was two
  64 MB device allocations for the soft-body heaps, about 2 ms each, straight after the
  narrow-phase launches. The narrow phase's output arrays came back as zeros.
- Everything seen earlier fits. A wait placed anywhere after those launches fixed it
  (10 of 10 runs); sleeping for 8 ms in the same place did not (0 of 10); neither did
  making pinned host memory coherent (`HIP_HOST_COHERENT=1`). Waiting on every kernel
  (`PHYSX_GPU_LAUNCH_SYNCHRONOUS=1`, `AMD_SERIALIZE_KERNEL=3`) fixed it because each wait
  flushes.
- Likely mechanism: the integrated GPU powers its graphics core down when idle (GFXOFF)
  and the caches do not survive that. **[assumed]** GFXOFF is enabled on the test machine
  (`amdgpu.ppfeaturemask` is `0xfff7bfff`, bit 15 set **[ran]**), and the amdgpu driver
  switches it off while compute queues are active only on GFX11 with older firmware and on
  GFX9 integrated GPUs ("to fix some compute applications issue on GFX9"); GFX10.3, which
  gfx1036 is, is not covered **[read: `amdgpu_amdkfd_set_compute_idle` in mainline Linux,
  2026-11-20]**. One observation fits: with a compute process attached but idle, the
  driver's dump of the hardware queue registers (`kfd/hqds` in debugfs) reads all ones,
  as a powered-down block does; while the process is working they read normally **[ran]**.
  The direct test,
  switching GFXOFF off through `/sys/kernel/debug/dri/0/amdgpu_gfxoff` and re-running the
  check, was tried and refused: with Secure Boot the kernel is in lockdown and rejects the
  write even from root. Still open: boot once with `amdgpu.ppfeaturemask=0xfff73fff` and
  re-run the check with `AMD_OPT_FLUSH=1`.
- Whether discrete Radeon cards lose data the same way is **unknown**. The runtime source
  already forces a system-scope acquire on gfx12 hardware, with the comment "needed for
  GFX12", so the device-scope default has needed correcting before. **[read]**

**Workaround in the port:** `PxHipHost.h` sets `AMD_OPT_FLUSH=0` in the process environment
before the first HIP call, unless the variable is already set. The runtime then flushes
after every kernel, which is what CUDA code takes for granted. With it:

- `SnippetDeformableVolumeKinematic` passes 12 of 12 runs (before: between 0 and 5 of 10);
- `SnippetDirectGPUAPIArticulation` passes 3 of 3;
- the comparison test passes 8 of 8 scenes with both solvers, and the snippet suite passes
  in full once the second defect below is worked around as well.

Limits of the workaround:

- The runtime reads the variable once, on the first HIP call in the process. A program that
  uses HIP itself before it creates the PhysX GPU context has to set `AMD_OPT_FLUSH=0` in
  its environment.
- It is applied on every GPU, because only one has been tested, and it costs time.
  Measured on the test GPU against the runtime's default: the eight rigid-body scenes of
  the comparison test are 1% to 14% slower per step; `SnippetPBF` takes 40.7 s instead of
  38.1 s, `SnippetPBFMultiMat` 8.5 s instead of 6.1 s, `SnippetIsosurface` 48.2 s instead
  of 34.9 s. If discrete cards turn out not to lose data, the setting should become
  specific to the GPUs that do.
- This is a defect in ROCm or the kernel driver, not in PhysX: a correct HIP program loses
  data. It has not been reported to AMD yet; the check program is a ready reproduction.

### A stream never runs once the process holds every compute queue (found, worked around)

`SnippetPBF` and `SnippetIsosurface` did not finish their first simulation step: both were
stopped after 25 minutes, with `AMD_OPT_FLUSH` at 0 and at 1. One thread sat in
`hipStreamSynchronize` at 100% CPU. **[ran]** unless marked:

- The HIP runtime backs streams with hardware queues. Each stream priority (low, normal,
  high) has its own pool, which grows to `GPU_MAX_HW_QUEUES` queues (4 by default), and
  the runtime keeps one more queue for itself. **[read]** (`rocclr/device/rocm/rocdevice.cpp`)
- PhysX creates one high-priority stream (broad phase), five or more at normal priority
  (solver, articulations, simulation core, narrow phase) and low-priority ones for
  particles, soft bodies and cloth. **[read]** The two snippets are the ones that use both
  particle streams, which takes the process to 1 + 1 + 4 + 2 = 8 compute queues.
- gfx1036 offers a process 8 compute queues (`num_cp_queues` in
  `/sys/class/kfd/kfd/topology/nodes/1/properties`).
- The driver's state during the stall (`kfd/rls` and `kfd/hqds` in debugfs, read as root):
  the run list holds the 8 compute queues; the GPU has 8 hardware slots for them (4 pipes
  with 2 queues each); 7 slots are active and 1 is empty. The queue without a slot has
  work that is never read. Four of the queues that do have a slot hold unread packets too;
  PhysX's streams wait on each other's events, so they are waiting for the one that cannot
  run. The driver does not count 8 queues as over-subscription; its "Runlist is getting
  oversubscribed" message appears from 9.
- Limiting the pools decides it:

  | `GPU_MAX_HW_QUEUES` | Compute queues `SnippetPBF` holds | Result |
  |---|---|---|
  | 1, 2 | 6 or fewer | passes, 40.3 s and 40.5 s |
  | 3 | 7 | passes, 40.8 s |
  | 4 (default) | 8 | stalls in the first step, every run |

- It is the process's own count that matters. `SnippetPBF` with 7 queues passed in 41 s
  beside a second process holding 4, 11 on the device in total, with the driver reporting
  over-subscription.
- **Not reproduced outside PhysX.** Standalone programs holding 8, 9, 10 and 13 queues
  finished, including one where every stream waits on an event of one other stream. So
  "8 queues" is necessary here but is not the whole trigger, and why the GPU leaves a slot
  empty is unknown.
- **Not explained:** on 2026-11-19 `SnippetPBF` passed in 279 s in this configuration, and
  on 2026-11-20 it stalled in every run. In between, the machine was rebooted and the GPU
  kernels were rebuilt after comment-only changes; the Linux kernel and ROCm packages are
  the same.

**Workaround in the port:** `PxHipHost.h` reads `num_cp_queues` for every GPU from sysfs
before the first HIP call and sets `GPU_MAX_HW_QUEUES` so that three pools plus the
runtime's own queue stay below the smallest count: 2 on the test GPU. A GPU with 14 or
more compute queues keeps the runtime's default, and a value the user has set is left
alone. With it `SnippetPBF` holds 6 compute queues and passes in 40.5 s;
`SnippetIsosurface` passes in 48.4 s.

Limits of the workaround:

- As with `AMD_OPT_FLUSH`, the runtime reads the variable once, on the first HIP call.
- The rule comes from one GPU. A program that creates HIP streams of its own next to PhysX
  adds to the count and can still reach the limit.
- No cost was measurable: the `SnippetPBF` times above are the same for 1, 2 and 3.

### Smaller items

- **LLVM 21 AMDGPU backend defect.** `convexCoreCollision.cu` fails at `-O2`/`-O3` on every
  RDNA target tried with "SOP2/SOPC instruction requires too many immediate constants".
  Built at `-O1` for that file only. To do: reduce and report; try AMD's LLVM (`clang-rocm`).
- The snippet renderer's CUDA/OpenGL buffer sharing is disabled under HIP.
- Only gfx1036 code objects are built by default; multi-target builds are untested.
- Windows is untouched.

## 9. Decisions

| Question | State |
|---|---|
| Where the port lives | `github.com/youhanasheriff/physx-hip`, public (2026-11-19) |
| Licence for the port's own files | Apache-2.0, the same as the SDK (2026-11-19) |
| Second test GPU | open; needed for milestone 2 (section 3) |

Nothing is pushed to, or opened against, NVIDIA's repository. Kit's closed binaries are not
reverse-engineered; the work uses only the published source.

## 10. Changes to upstream files

Source and build changes are guarded so that the CUDA build is unchanged; each carries a
`Modified for the HIP port` comment. The CUDA build itself has not been re-run (no CUDA
toolkit on the test machine). **[assumed intact]**

| File | Change |
|---|---|
| `README.md` | notice and status of the port added above NVIDIA's text, which is unchanged |
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

- `github.com/eugenesflanagin-spec/physx5-zluda` (Oct 2026): PhysX 5.11 built PTX-only and
  run through ZLUDA on an RX 9070 XT with an `LD_PRELOAD` shim. Its notes report 12 of 488
  kernels failing (function pointers, in convexCore and particles) and trajectories that
  agree qualitatively.
- SCALE (Spectral Compute) lists PhysX in its validation suite as tested and not passing;
  its build stops at the PTX `bar` instructions. gfx1036 is not a SCALE target.

ZLUDA's own PhysX work targets 32-bit legacy game PhysX. Isaac Sim skips non-NVIDIA GPUs.
The search was one pass on 2026-11-19; Reddit and Hacker News were not searched in depth.

## 12. Reproducing

```sh
# stock CPU build and tests
cd physx && ./generate_projects.sh linux-clang-cpu-only
cd compiler/linux-clang-cpu-only-checked
cmake -DPX_WARNINGS_AS_ERRORS=OFF -DPX_SNIPPETS_HEADLESS=ON . && make -j"$(nproc)"
../../../hip/scripts/run_snippets.sh ../../bin/linux.x86_64/checked

# HIP build (overwrites the same bin directory)
cd physx && ./generate_projects.sh linux-clang-hip
make -C compiler/linux-clang-hip-checked -j"$(nproc)"

# tests against the HIP build
hip/probe/run.sh                                        # runtime primitives
hip/tests/build.sh && physx/bin/linux.x86_64/checked/cpu_gpu_compare [--tgs]
hip/scripts/run_snippets.sh physx/bin/linux.x86_64/checked

# debugging aids
PHYSX_GPU_LAUNCH_SYNCHRONOUS=1 <program>   # PhysX waits on every kernel launch
AMD_SERIALIZE_KERNEL=3 <program>            # HIP waits on every kernel
AMD_LOG_LEVEL=3 <program>                   # HIP API trace (4 adds every queue packet)
AMD_OPT_FLUSH=1 <program>                   # the runtime's default flushing, which the GPU library turns off (section 8)
GPU_MAX_HW_QUEUES=4 <program>               # the runtime's default queue pools, which the GPU library caps on small GPUs (section 8)
```

The extra checks in `hip/probe` (`check_*.cpp`) are separate programs; each has its build
and run line at the top of the file. They use the kernels `hip/probe/run.sh` builds.

Note: the CPU-only and HIP presets write to the same `physx/bin/linux.x86_64/<config>`
directory, so building one replaces the other's binaries.
