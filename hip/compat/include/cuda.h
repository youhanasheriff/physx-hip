// Compatibility header for building PhysX GPU code with HIP.
// This file is part of the HIP port and is not part of upstream PhysX.
//
// Stands in for the CUDA driver API header. Device code gets the HIP kernel language;
// host code gets the cu* driver API mapped onto HIP (PxHipDriver.h).

#ifndef PX_HIP_COMPAT_CUDA_H
#define PX_HIP_COMPAT_CUDA_H

#include "PxHipDevice.h"

#endif
