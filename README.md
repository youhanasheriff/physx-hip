# PhysX on AMD GPUs (HIP port)

An unofficial port of the GPU side of [NVIDIA PhysX](https://github.com/NVIDIA-Omniverse/PhysX)
from CUDA to AMD [HIP](https://github.com/ROCm/HIP) (ROCm), so that PhysX's GPU pipeline can
run on Radeon hardware.

**This is work in progress and not ready for production use.** It is not affiliated with,
endorsed by or supported by NVIDIA or AMD. PhysX and Omniverse are trademarks of NVIDIA
Corporation. Report problems with the port in this repository's issue tracker, not to NVIDIA.

## Where it stands

Tested on one GPU so far: the integrated Radeon of a Ryzen 9 9950X (`gfx1036`), Ubuntu 26.04,
ROCm 7.1.

| | |
|---|---|
| PhysX SDK 5.13.0 builds with a HIP backend; all 61 kernel sources (491 kernels) compile and load | works |
| Rigid bodies on the GPU pipeline, compared against the CPU pipeline | 8 of 8 scenes agree, PGS and TGS |
| GPU snippets (particles, deformables, SDF, direct GPU API) | 12 of 14 run clean; results not yet validated numerically |
| `SnippetDeformableVolumeKinematic`, `SnippetDirectGPUAPIArticulation` | crash: an open ordering race |
| Discrete Radeon cards, performance, Windows | untested |
| ovphysx and the omni.physx Kit extension | not started |

Only GPUs with 32-lane wavefronts are supported: Radeon RX 5000 series and later, and Ryzen
integrated graphics. AMD Instinct accelerators run 64-lane wavefronts and are out of scope.

[PORTING.md](PORTING.md) has the full record: what was verified and how, the design, the
NVIDIA-specific behaviours PhysX depends on, open defects, and the milestone plan.

## Building

Needs a Linux machine with ROCm 7.1 or later (`hipcc`, `libamdhip64-dev`, `rocminfo`),
clang, and a user in the `render` and `video` groups.

```sh
cd physx
./generate_projects.sh linux-clang-hip
make -C compiler/linux-clang-hip-checked -j"$(nproc)"
cd ..

hip/probe/run.sh                                   # checks the GPU and HIP runtime
hip/tests/build.sh && physx/bin/linux.x86_64/checked/cpu_gpu_compare
hip/scripts/run_snippets.sh physx/bin/linux.x86_64/checked
```

The CUDA build is still selected by default (`PX_GPU_BACKEND=CUDA`) and is meant to be
unchanged by the port.

## What is in the port

- `hip/`: everything new. Stand-in headers that map the CUDA vocabulary onto HIP, the CMake
  backend that compiles and embeds the kernels, an LLVM pass that reproduces NVIDIA's shift
  behaviour, a capability probe, and tests.
- A small number of guarded edits to NVIDIA's sources, each marked
  `Modified for the HIP port` and listed in [PORTING.md](PORTING.md).

## Licence

NVIDIA's code keeps its licences and notices: BSD-3-Clause at the root, Apache-2.0 for the
PhysX SDK (`physx/LICENSE.md`) and ovphysx. The port's own files under `hip/` and its
changes to the SDK are Apache-2.0 ([hip/LICENSE](hip/LICENSE)).

The base is upstream tag `ovphysx-0.8.0`. NVIDIA's original README follows unchanged.

---

# NVIDIA PhysX

<details>
<summary>Copyright & License</summary>

Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

</details>

Please also see license files in the root folder and in the respective subfolders.

## Content

This repo contains:

| Directory | Description |
|---|---|
| [`ovphysx/`](ovphysx/) | ovphysx — C API with Python bindings for USD physics simulation with DLPack tensor interop (`pip install ovphysx`) |
| [`physx/`](physx/) | PhysX SDK — real-time physics simulation engine |

### ovphysx

| Release | Compatibility |
|---|---|
| [0.8](https://github.com/NVIDIA-Omniverse/PhysX/tree/ovphysx-0.8.0) | PhysX SDK 5.13.0 |
| [0.7](https://github.com/NVIDIA-Omniverse/PhysX/tree/ovphysx-0.7.0) | PhysX SDK 5.12.0 |
| [0.6](https://github.com/NVIDIA-Omniverse/PhysX/tree/ovphysx-0.6.4) | PhysX SDK 5.11.0 |
| [0.5](https://github.com/NVIDIA-Omniverse/PhysX/tree/ovphysx-0.5.11) | PhysX SDK 5.10.0 |
| [0.4](https://github.com/NVIDIA-Omniverse/PhysX/tree/ovphysx-0.4.13) | PhysX SDK 5.9.0 |
| [0.3](https://github.com/NVIDIA-Omniverse/PhysX/tree/ovphysx-0.3.7) | PhysX SDK 5.9.0 |
| [0.2](https://github.com/NVIDIA-Omniverse/PhysX/tree/ovphysx-0.2.9) | PhysX SDK 5.8.0 |

### Omniverse PhysX Extensions and PhysX SDK

| Release | Compatibility |
|---|---|
| [110.1 and 5.9.0](https://github.com/NVIDIA-Omniverse/PhysX/tree/110.1-omni-and-physx-5.9.0) | [IsaacSim 6.0.0](https://github.com/isaac-sim/IsaacSim/tree/v6.0.0) |
| [110.0 and 5.8.0](https://github.com/NVIDIA-Omniverse/PhysX/tree/110.0-omni-and-physx-5.8.0) | [IsaacSim 6.0.0-dev2](https://github.com/isaac-sim/IsaacSim/tree/v6.0.0-dev2) |
| [109.0 and 5.7.0](https://github.com/NVIDIA-Omniverse/PhysX/tree/109.0-omni-and-physx-5.7.0) | [IsaacSim 6.0.0-dev](https://github.com/isaac-sim/IsaacSim/tree/v6.0.0-dev) |
| [107.3 and 5.6.1](https://github.com/NVIDIA-Omniverse/PhysX/tree/107.3-omni-and-physx-5.6.1) | [IsaacSim 5.1.0](https://github.com/isaac-sim/IsaacSim/tree/v5.1.0) |

Additional simulation libraries:

| Directory | Description |
|---|---|
| [`blast/`](blast/) | Blast SDK — destruction and fracture simulation |
| [`flow/`](flow/) | Flow SDK — fluid and fire simulation |

## Support

* Please use GitHub [Discussions](https://github.com/NVIDIA-Omniverse/PhysX/discussions/) for questions and comments.
* GitHub [Issues](https://github.com/NVIDIA-Omniverse/PhysX/issues) should only be used for bug reports or documentation issues.
* You can also ask questions in the NVIDIA Omniverse #physics [Discord Channel](https://discord.com/invite/XWQNJDNuaC).


<!-- physics-component-licenses-begin -->
## Component licenses

These licenses apply to the listed components; other components retain their own licenses.

- `ovphysx/` source: [Apache License 2.0](ovphysx/LICENSE.txt), with the bundled-component exceptions in [third-party notices](ovphysx/THIRD_PARTY_NOTICES.md).
- `physx/` source: [Apache License 2.0](physx/LICENSE.md), with the bundled-component exceptions in [third-party notices](physx/THIRD_PARTY_NOTICES.md).
- ovphysx binaries: [NVIDIA Omniverse agreements](ovphysx/licenses/LICENSE-binary.txt).
- Omniverse Extensions dependencies: [NVIDIA agreements](ovphysx/tools/internal-licenses/Omniverse-Extensions-LICENSE.txt).
- OpenUSD-derived code: [Tomorrow Open Source Technology License 1.0](ovphysx/tools/internal-licenses/OpenUSD-LICENSE.txt).
- DLPack: [Apache License 2.0, upstream contributor attribution](ovphysx/tools/internal-licenses/dlpack-LICENSE.txt).
- Repoman bootstrap tooling: [MIT](ovphysx/tools/internal-licenses/repoman-LICENSE.txt).
- V-HACD code: [BSD 3-Clause](ovphysx/tools/internal-licenses/vhacd-LICENSE.txt).
<!-- physics-component-licenses-end -->
