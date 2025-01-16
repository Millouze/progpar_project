#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <driver_types.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "SimulationNBodyCUDA.hpp"
#include "core/Bodies.hpp"

// static dim3 blocksPerGrid = {1};
// static dim3 threadsPerBlock = {1024};
// static float *d_qx, *d_qy, *d_qz, *d_m;
// static accAoS_t<float> *d_accelerations;

//static dim3 blocksPerGrid = {60} ;
//static dim3 threadsPerBlock = {1024};

__constant__ float softSquared;
__constant__ float gravity;


SimulationNBodyCUDA::SimulationNBodyCUDA(const unsigned long nBodies, const std::string &scheme, const float soft,
                                         const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{

    const float hsoft = this->soft * this->soft;
    
    cudaMemcpyToSymbol(softSquared, &hsoft, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gravity, &this->G, sizeof(float), 0, cudaMemcpyHostToDevice);

    const unsigned long arraySize = sizeof(float) * nBodies;
    const unsigned long accSize = sizeof(struct accAoS_t<float>) * nBodies;
    this->flopsPerIte = 19.f * (float)this->getBodies().getN() * (float)this->getBodies().getN() + 3.f * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
    cudaMalloc(&this->d_qx, arraySize);
    cudaMalloc(&this->d_qy, arraySize);
    cudaMalloc(&this->d_qz, arraySize);
    cudaMalloc(&this->d_m, arraySize);
    cudaMalloc(&this->d_accelerations, accSize);
}

 SimulationNBodyCUDA::~SimulationNBodyCUDA() {
      cudaFree(d_qx);
      cudaFree(d_qy);
      cudaFree(d_qz);
      cudaFree(d_m);
      cudaFree(d_accelerations);
    }


void SimulationNBodyCUDA::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

static __global__ void computeBodiesAcceleration(const unsigned long nBodies,
                                          float *qx, float *qy, float *qz, float *m, accAoS_t<float> *accelerations)
{
        int x = blockDim.x * blockIdx.x + threadIdx.x;

        if(x > nBodies){
            return;
        }
        
        
        float ax = 0, ay = 0, az = 0;
        for (unsigned long jBody = 0; jBody < nBodies; jBody++) {

            // All forces of bodies of indexes lower than the current one have already been added to current body's
            // accel skiping.
            const float rijx = qx[jBody]- qx[x]; // 1 flop
            const float rijy = qy[jBody]- qy[x]; // 1 flop
            const float rijz = qz[jBody] - qz[x]; // 1 flop

            // compute the || rij ||² distance between body i and body j
            float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            // compute e²
            rijSquared += softSquared; // 1 flop

            const float pow = rsqrtf(rijSquared); // 1 flops

            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = m[jBody] * (pow * pow * pow); // 3 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            ax += ai * rijx; // 2 flops
            ay += ai * rijy; // 2 flops
            az += ai * rijz; // 2 flops
            
            // accelerations[x].ax += ai *rijx;            
            // accelerations[x].ay += ai * rijy;            
            // accelerations[x].az += ai * rijz;            

            // Adding acceleration forces to the j body as well.
        }

        accelerations[x].ax = ax * gravity;
        accelerations[x].ay = ay * gravity;
        accelerations[x].az = az * gravity;
}

void SimulationNBodyCUDA::computeOneIteration()
{
    this->initIteration();
    //const float softSquared = this->soft * this->soft;
    const unsigned long nBodies = this->getBodies().getN();
    dataSoA_t<float> h_bodies = this->getBodies().getDataSoA();
    const unsigned long arraySize = sizeof(float) * nBodies;
    const unsigned long accSize = sizeof(struct accAoS_t<float>) * nBodies;

    // cudaMalloc(&this->d_qx, arraySize);
    // cudaMalloc(&this->d_qy, arraySize);
    // cudaMalloc(&this->d_qz, arraySize);
    // cudaMalloc(&this->d_m, arraySize);
    // cudaMalloc(&this->d_accelerations, accSize);
    cudaMemset(d_accelerations, 0, accSize);
    cudaMemset(d_qx, 0, arraySize);
    cudaMemset(d_qy, 0, arraySize);
    cudaMemset(d_qz, 0, arraySize);
    cudaMemset(d_m, 0, arraySize);
    

    cudaMemcpy(d_qx, &(h_bodies.qx[0]), arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, &(h_bodies.qy[0]), arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, &(h_bodies.qz[0]), arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &(h_bodies.m[0]), arraySize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = {512};
    dim3 blocksPerGrid = {(nBodies+threadsPerBlock.x -1)/threadsPerBlock.x};
    

    computeBodiesAcceleration<<< blocksPerGrid,threadsPerBlock >>>(nBodies, d_qx, d_qy, d_qz, d_m, d_accelerations);

    cudaDeviceSynchronize();

    cudaMemcpy(this->accelerations.data(), d_accelerations, accSize, cudaMemcpyDeviceToHost);

    // cudaFree(d_qx);
    // cudaFree(d_qy);
    // cudaFree(d_qz);
    // cudaFree(d_m);
    // cudaFree(d_accelerations); 
    
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
