#ifndef COMPUTE_H
#define COMPUTE_H

#include "vector.h"

void initDeviceMemory(int numEntities);
void freeDeviceMemory();
void copyHostToDevice();
void copyDeviceToHost();
#ifdef __cplusplus
extern "C" {
    #endif
    void compute();
    #ifdef __cplusplus
}
#endif

#endif // COMPUTE_H
