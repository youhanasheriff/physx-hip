// Compatibility header for building PhysX GPU code with HIP.
// This file is part of the HIP port and is not part of upstream PhysX.
//
// Stands in for CUDA's texture_types.h.

#ifndef PX_HIP_COMPAT_TEXTURE_TYPES_H
#define PX_HIP_COMPAT_TEXTURE_TYPES_H

#if defined(__HIPCC__)
#include "PxHipDevice.h"
#else
#include "PxHipHost.h"
#include "PxHipVectorTypes.h"
#endif

#endif
