## SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: Apache-2.0

#
# Build SnippetVehicle linux template
#

SET(SNIPPET_PLATFORM_INCLUDES
)

# Modified for the HIP port (not part of upstream PhysX): PX_SNIPPETS_HEADLESS builds the
# snippets without RENDER_SNIPPET so they run a fixed number of steps and exit, which lets
# them be used as tests on a machine with no display. The default keeps upstream behaviour.
OPTION(PX_SNIPPETS_HEADLESS "Build snippets without the interactive render loop" OFF)
IF(PX_SNIPPETS_HEADLESS)
	SET(PX_SNIPPET_RENDER_DEF "")
ELSE()
	SET(PX_SNIPPET_RENDER_DEF "RENDER_SNIPPET")
ENDIF()

SET(SNIPPET_COMPILE_DEFS
	# Common to all configurations

	${PHYSX_LINUX_COMPILE_DEFS};${PX_SNIPPET_RENDER_DEF}

	$<$<CONFIG:debug>:${PHYSX_LINUX_DEBUG_COMPILE_DEFS};>
	$<$<CONFIG:checked>:${PHYSX_LINUX_CHECKED_COMPILE_DEFS};>
	$<$<CONFIG:profile>:${PHYSX_LINUX_PROFILE_COMPILE_DEFS};>
	$<$<CONFIG:release>:${PHYSX_LINUX_RELEASE_COMPILE_DEFS};>
)

SET(SNIPPET_PLATFORM_SOURCES
	${PHYSX_ROOT_DIR}/snippets/snippetcommon/ClassicMain.cpp
)

SET(SNIPPET_PLATFORM_LINKED_LIBS SnippetRender OpenGL::GL OpenGL::GLU GLUT::GLUT X11 rt pthread dl)

IF(PX_GENERATE_GPU_STATIC_LIBRARIES)
	LIST(APPEND SNIPPET_PLATFORM_LINKED_LIBS PhysXGpu)
ENDIF()

IF(NOT PX_GENERATE_STATIC_LIBRARIES)
	LIST(APPEND SNIPPET_PLATFORM_LINKED_LIBS PhysXTask)
ENDIF()
