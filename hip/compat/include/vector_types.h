// Compatibility header for building PhysX GPU code with HIP.
// This file is part of the HIP port and is not part of upstream PhysX.
//
// Stands in for CUDA's vector_types.h (float4, uint4, ...).

#if defined(__HIPCC__)
#include "PxHipDevice.h"
#else
#include "PxHipVectorTypes.h"
#endif
