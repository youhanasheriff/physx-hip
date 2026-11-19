// Copyright (c) 2026 Youhana Sheriff
// SPDX-License-Identifier: Apache-2.0
//
// Compatibility header for building PhysX GPU code with HIP.
// This file is part of the HIP port and is not part of upstream PhysX.
//
// Stands in for CUDA's driver_types.h.

#ifndef PX_HIP_COMPAT_DRIVER_TYPES_H
#define PX_HIP_COMPAT_DRIVER_TYPES_H

#if defined(__HIPCC__)
#include "PxHipDevice.h"
#else
#include "PxHipHost.h"
#include "PxHipVectorTypes.h"
#endif

#endif
