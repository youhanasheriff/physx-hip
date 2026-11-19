// Compatibility header for building PhysX GPU code with HIP.
// This file is part of the HIP port and is not part of upstream PhysX.
//
// Stands in for CUDA's vector_functions.h (make_float4, ...).

#ifndef PX_HIP_COMPAT_VECTOR_FUNCTIONS_H
#define PX_HIP_COMPAT_VECTOR_FUNCTIONS_H

#if defined(__HIPCC__)
#include "PxHipDevice.h"
#else
#include "PxHipHost.h"
#include "PxHipVectorTypes.h"
#endif

#endif
