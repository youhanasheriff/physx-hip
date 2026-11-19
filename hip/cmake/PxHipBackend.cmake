# HIP backend for the PhysX GPU build.
#
# This file is part of the HIP port and is not part of upstream PhysX.
#
# Included by the PhysX build when PX_GPU_BACKEND is HIP. Instead of enabling CMake's CUDA
# language, each kernel source (.cu) is compiled for AMD GPUs with clang in HIP mode,
# device code only, into a code object. The code objects are embedded in the GPU library
# as data and registered with PhysX's module table at load time, which is what nvcc's
# generated static initialisers do in the CUDA build.
#
# Provides:
#   px_hip_setup_target(<target>)             host code of <target> sees the CUDA stand-in headers
#   px_hip_add_kernels(<target> <file.cu>...) compile, embed and register kernels for <target>

GET_FILENAME_COMPONENT(PX_HIP_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
SET(PX_HIP_COMPAT_INCLUDE_DIR "${PX_HIP_ROOT}/compat/include")

FIND_PACKAGE(hip REQUIRED CONFIG)

# Device compiler: the clang that the installed HIP is set up for.
IF(NOT PX_HIP_CLANG)
	FIND_PROGRAM(PX_HIPCONFIG hipconfig)
	IF(PX_HIPCONFIG)
		EXECUTE_PROCESS(COMMAND ${PX_HIPCONFIG} --hipclangpath OUTPUT_VARIABLE _px_hip_clang_dir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
	ENDIF()
	FIND_PROGRAM(PX_HIP_CLANG clang++ HINTS "${_px_hip_clang_dir}" REQUIRED)
ENDIF()

# GPU architectures to build kernels for. PhysX's kernels assume 32-thread warps, so only
# targets that run 32-lane wavefronts are valid (RDNA: gfx10xx, gfx11xx, gfx12xx).
SET(PX_HIP_ARCHITECTURES "" CACHE STRING "AMD GPU architectures to build kernels for, e.g. gfx1036;gfx1100. Empty: the GPUs found in this machine.")
IF(NOT PX_HIP_ARCHITECTURES)
	FIND_PROGRAM(PX_ROCMINFO rocminfo)
	IF(PX_ROCMINFO)
		EXECUTE_PROCESS(COMMAND ${PX_ROCMINFO} OUTPUT_VARIABLE _px_rocminfo ERROR_QUIET)
		STRING(REGEX MATCHALL "Name: +gfx[0-9a-f]+" _px_gfx_lines "${_px_rocminfo}")
		SET(_px_archs "")
		FOREACH(_line ${_px_gfx_lines})
			STRING(REGEX REPLACE "Name: +" "" _arch "${_line}")
			LIST(APPEND _px_archs ${_arch})
		ENDFOREACH()
		LIST(REMOVE_DUPLICATES _px_archs)
	ENDIF()
	IF(NOT _px_archs)
		MESSAGE(FATAL_ERROR "No AMD GPU found by rocminfo. Set PX_HIP_ARCHITECTURES, e.g. -DPX_HIP_ARCHITECTURES=gfx1100")
	ENDIF()
	SET(PX_HIP_ARCHITECTURES "${_px_archs}" CACHE STRING "AMD GPU architectures to build kernels for" FORCE)
ENDIF()
FOREACH(_arch ${PX_HIP_ARCHITECTURES})
	IF(_arch MATCHES "^gfx[89]")
		MESSAGE(FATAL_ERROR "${_arch} runs 64-lane wavefronts; PhysX kernels need 32-lane wavefronts (gfx10xx and later Radeon parts).")
	ENDIF()
ENDFOREACH()
STRING(REPLACE ";" "," _px_hip_arch_csv "${PX_HIP_ARCHITECTURES}")
MESSAGE(STATUS "PhysX GPU backend: HIP ${hip_VERSION}, device compiler ${PX_HIP_CLANG}, architectures ${_px_hip_arch_csv}")

# Compiler pass that gives kernels PTX's behaviour for oversized shifts; see the source file.
# It is an LLVM plugin and has to be built against the LLVM that the device compiler uses.
GET_FILENAME_COMPONENT(_px_hip_clang_bin "${PX_HIP_CLANG}" DIRECTORY)
FIND_PROGRAM(PX_HIP_LLVM_CONFIG llvm-config HINTS "${_px_hip_clang_bin}" NO_DEFAULT_PATH REQUIRED)
EXECUTE_PROCESS(COMMAND ${PX_HIP_LLVM_CONFIG} --cxxflags OUTPUT_VARIABLE _px_llvm_cxxflags OUTPUT_STRIP_TRAILING_WHITESPACE)
SEPARATE_ARGUMENTS(_px_llvm_cxxflags)
SET(PX_HIP_PTX_PLUGIN "${CMAKE_BINARY_DIR}/hip_tools/libPxPtxSemantics.so" CACHE INTERNAL "PTX semantics pass plugin")
ADD_CUSTOM_COMMAND(
	OUTPUT "${PX_HIP_PTX_PLUGIN}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/hip_tools"
	COMMAND ${PX_HIP_CLANG} -shared -fPIC -O2 ${_px_llvm_cxxflags} "${PX_HIP_ROOT}/ptx-semantics/PxPtxSemantics.cpp" -o "${PX_HIP_PTX_PLUGIN}"
	DEPENDS "${PX_HIP_ROOT}/ptx-semantics/PxPtxSemantics.cpp"
	COMMENT "Building PTX semantics compiler pass"
	VERBATIM
)
ADD_CUSTOM_TARGET(PxPtxSemanticsPlugin DEPENDS "${PX_HIP_PTX_PLUGIN}")

# nvcc's -use_fast_math contracts multiply-adds and relaxes division and square root, but
# does not assume finite maths; clang's -ffast-math would, and PhysX tests for NaN and
# infinity. Only contraction is enabled here.
# The kernels cast freely between unrelated pointer types (float4, PxVec3, raw bytes), which
# nvcc tolerates; clang's type-based alias analysis must be off for them.
SET(PX_HIP_KERNEL_FLAGS
	-x hip --cuda-device-only --offload-arch=${_px_hip_arch_csv} -mno-wavefrontsize64
	-std=c++14 -ffp-contract=fast -fno-strict-aliasing -fno-rtti -fno-exceptions -fvisibility=hidden -w
	-include PxHipDevice.h -fpass-plugin=${PX_HIP_PTX_PLUGIN}
	CACHE INTERNAL "Compiler flags for PhysX kernels under HIP")

# Kernel sources that hit a compiler defect at -O3, and the level that works.
# convexCoreCollision: LLVM 21 AMDGPU backend, "SOP2/SOPC instruction requires too many immediate constants" at -O2 and -O3.
SET(PX_HIP_KERNEL_OPT_convexCoreCollision -O1)

FUNCTION(px_hip_setup_target target)
	TARGET_INCLUDE_DIRECTORIES(${target} BEFORE PRIVATE ${PX_HIP_COMPAT_INCLUDE_DIR})
	TARGET_COMPILE_DEFINITIONS(${target} PRIVATE __HIP_PLATFORM_AMD__)
ENDFUNCTION()

FUNCTION(px_hip_add_kernels target)
	px_hip_setup_target(${target})

	SET(_out_dir "${CMAKE_CURRENT_BINARY_DIR}/hip_kernels/${target}")
	SET(_incs "$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>")
	SET(_defs "$<FILTER:$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>,EXCLUDE,^$>")

	SET(_code_objects "")
	SET(_asm "")
	SET(_decls "")
	SET(_regs "")
	SET(_stubs "")
	FOREACH(_cu ${ARGN})
		GET_FILENAME_COMPONENT(_name "${_cu}" NAME_WE)
		SET(_co "${_out_dir}/${_name}.co")
		SET(_opt -O3)
		IF(DEFINED PX_HIP_KERNEL_OPT_${_name})
			SET(_opt ${PX_HIP_KERNEL_OPT_${_name}})
		ENDIF()
		ADD_CUSTOM_COMMAND(
			OUTPUT "${_co}"
			COMMAND ${CMAKE_COMMAND} -E make_directory "${_out_dir}"
			COMMAND ${PX_HIP_CLANG} ${PX_HIP_KERNEL_FLAGS} ${_opt}
				"$<$<BOOL:${_defs}>:-D$<JOIN:${_defs},;-D>>"
				"$<$<BOOL:${_incs}>:-I$<JOIN:${_incs},;-I>>"
				-MD -MF "${_co}.d" -MT "${_co}"
				"${_cu}" -o "${_co}"
			DEPENDS "${_cu}" "${PX_HIP_COMPAT_INCLUDE_DIR}/PxHipDevice.h" "${PX_HIP_PTX_PLUGIN}"
			DEPFILE "${_co}.d"
			COMMENT "Building HIP kernels ${_name}.cu"
			COMMAND_EXPAND_LISTS VERBATIM
		)
		LIST(APPEND _code_objects "${_co}")

		# Each kernel source defines an empty host function that host code calls to force the
		# object file to be linked. The host never compiles the kernel sources under HIP, so
		# the generated file defines those functions instead.
		FILE(STRINGS "${_cu}" _host_stubs REGEX "extern \"C\" __host__ void init[A-Za-z0-9_]+\\(\\)")
		FOREACH(_stub ${_host_stubs})
			STRING(REGEX MATCH "init[A-Za-z0-9_]+" _stub_name "${_stub}")
			STRING(APPEND _stubs "extern \"C\" void ${_stub_name}() {}\n")
		ENDFOREACH()
		SET_PROPERTY(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_cu}")
		STRING(APPEND _asm "\t\".balign 16\\n\"\n\t\"pxHipKernels_${_name}:\\n\"\n\t\".incbin \\\"${_co}\\\"\\n\"\n")
		STRING(APPEND _decls "extern \"C\" const char pxHipKernels_${_name}[];\n")
		STRING(APPEND _regs "\t\tPxGpuCudaRegisterFatBinary(const_cast<char*>(pxHipKernels_${_name}));\n")
	ENDFOREACH()

	SET(_gen "${_out_dir}/${target}_hip_kernels.cpp")
	FILE(CONFIGURE OUTPUT "${_gen}" @ONLY CONTENT
"// Generated by PxHipBackend.cmake. Embeds the HIP kernel code objects of ${target}
// and registers them with PhysX's GPU module table when the library is loaded.

#ifndef PX_PHYSX_GPU_STATIC
	#define PX_PHYSX_GPU_STATIC
#endif
#include \"PxPhysXGpu.h\"

__asm__(
	\".pushsection .rodata.pxhip,\\\"a\\\",@progbits\\n\"
${_asm}	\".popsection\\n\"
);

${_decls}
${_stubs}
namespace
{
	struct PxHipKernelRegistration
	{
		PxHipKernelRegistration()
		{
${_regs}		}
	};
	const PxHipKernelRegistration gPxHipKernelRegistration;
}
")
	TARGET_SOURCES(${target} PRIVATE "${_gen}")
	ADD_DEPENDENCIES(${target} PxPtxSemanticsPlugin)
	SET_SOURCE_FILES_PROPERTIES("${_gen}" PROPERTIES OBJECT_DEPENDS "${_code_objects}" COMPILE_OPTIONS "-w")
ENDFUNCTION()
