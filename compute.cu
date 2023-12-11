#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
//include "book.h" //HANDLE_ERROR
//include compute (issues with compute in nbody)
#include "compute.h"
//cuda include lib
#include <cuda_runtime.h>

/*
#define HANDLE_ERROR(err)
    do {
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);
        }
    } while (0)
    handle error instead of lib inclusion, giving errors so not include in cuda lines
*/

#define BLOCK_SIZE 256 //max num of threads per GPU in mainstream architecture is 512 so choose between 128/256/512

//globals CUDA kernels (see lecture 21 slide 7 and lecture 20)
//numEntities -> passed in NUMENTITIES
//compute acceleration between things - should compute dot product
__global__ void accel(double *hPos, double *mass, double *accels, int numEntities) {
    //printf("accel why");
    //get thread index
    //int thIdx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = blockIdx.x + blockIdx.y * blockDim.x;
    //index in entities?
    if (i < numEntities) {
        for (int j = 0; j < numEntities; j++) {
            if (i != j) {
                //distance between i and j (objects)
                double distance[3];
                for (int k = 0; k < 3; k++) {
                    distance[k] = hPos[i * 3 + k] - hPos[j * 3 + k];
                }
                //constants
                double magSq = (distance[0] * distance[0]) + (distance[1] * distance[1]) + (distance[2] * distance[2]);
                double mag = sqrt(magSq);
                double accelmag = -GRAV_CONSTANT * mass[j] / magSq;
                //accelrations
                accels[i * numEntities + j] = accelmag * distance[0] / mag;
                accels[i * numEntities + j + numEntities] = accelmag * distance[1] / mag;
                accels[i * numEntities + j + 2 * numEntities] = accelmag * distance[2] / mag;
            } else {
                //i and j objects r the same so no accelration
                accels[i * numEntities + j] = 0.0;
            }
        }
    }
}

//sum columns in acceleration array
__global__ void colSum(double *accels, double *accelSum, int numEntities) {
    //printf("colSum");
    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.x + blockIdx.y * blockDim.x;
    if (i < numEntities) {
        accelSum[i] = 0.0;
        //__synchthreads();?
        for (int j = 0; j < numEntities; j++) {
            accelSum[i] += accels[j * numEntities + i];
        }
    }
}

//update velocity and position
__global__ void update(double *hVel, double *hPos, double *accelSum, int numEntities, double dt) {
    //printf("I'm in update");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numEntities) {
        for (int k = 0; k < 3; k++) {
            hVel[i * 3 + k] += accelSum[i * 3 + k] * dt;
            hPos[i * 3 + k] += hVel[i * 3 + k] * dt;
        }
    }
}

// compute: Updates the positions and locations of the objects in the system based on gravity using GPU
void compute() {
    //printf("in compute first");
    //changed function - delete previous
    //device mem and kernel variables
    double *dev_hPos, *dev_mass, *dev_accels, *dev_accelSum;
	double *dev_hVel;
    
    //malloc
	cudaMalloc((void **)&dev_hVel, sizeof(double) * 3 * NUMENTITIES);
    cudaMalloc((void **)&dev_hPos, sizeof(double) * 3 * NUMENTITIES);
    cudaMalloc((void **)&dev_mass, sizeof(double) * NUMENTITIES);
    cudaMalloc((void **)&dev_accels, sizeof(double) * NUMENTITIES * NUMENTITIES * 3);
    cudaMalloc((void **)&dev_accelSum, sizeof(double) * NUMENTITIES * 3);

    //host to device
    cudaMemcpy(dev_hPos, hPos, sizeof(double) * 3 * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_hVel, hVel, sizeof(double) * 3 * NUMENTITIES, cudaMemcpyHostToDevice);

    int numBlocks = (NUMENTITIES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    //kernel calls
    accel<<<numBlocks, BLOCK_SIZE>>>(dev_hPos, dev_mass, dev_accels, NUMENTITIES);
    colSum<<<numBlocks, BLOCK_SIZE>>>(dev_accels, dev_accelSum, NUMENTITIES);
    update<<<numBlocks, BLOCK_SIZE>>>(dev_hVel, dev_hPos, dev_accelSum, NUMENTITIES, INTERVAL);

    //device to host
    cudaMemcpy(hVel, dev_hVel, sizeof(double) * 3 * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos, dev_hPos, sizeof(double) * 3 * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dev_hVel, sizeof(double) * 3 * NUMENTITIES, cudaMemcpyDeviceToHost);

    //free mem
    cudaFree(dev_hPos);
    cudaFree(dev_mass);
    cudaFree(dev_accels);
    cudaFree(dev_accelSum);
	cudaFree(dev_hVel);
    //free mem from CPU
    //free(hPos);
    //free(hVel);
    //free(mass);
    //printf("In compute last");
}
