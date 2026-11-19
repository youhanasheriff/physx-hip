// Compatibility header for building PhysX GPU code with HIP.
// This file is part of the HIP port and is not part of upstream PhysX.
//
// Stands in for the CUDA driver API header. Kernel sources get the HIP kernel language
// (PxHipDevice.h); host code gets the cu* driver API mapped onto HIP (PxHipHost.h).

#ifndef PX_HIP_COMPAT_CUDA_H
#define PX_HIP_COMPAT_CUDA_H

#if defined(__HIPCC__)
#include "PxHipDevice.h"
#else
#include "PxHipHost.h"
#include "PxHipVectorTypes.h"
#endif

#endif
