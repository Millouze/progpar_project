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

static dim3 blocksPerGrid = {1};
static dim3 threadsPerBlock = {1024};
static float *d_qx, *d_qy, *d_qz, *d_m;
static accAoS_t<float> *d_accelerations;


SimulationNBodyCUDA::SimulationNBodyCUDA(const unsigned long nBodies, const std::string &scheme, const float soft,
                                         const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyCUDA::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

__global__ void computeBodiesAcceleration(const unsigned long nBodies, const float softSquared, const float G,
                                          float *qx, float *qy, float *qz, float *m, accAoS_t<float> *accelerations)
{
        int x = blockDim.x * blockIdx.x + threadIdx.x;

        if(x == 0){
            // printf("\n");
            // printf("qx : %f\n", *(qx+sizeof(float))*10000);
            // printf("\n");
            // printf("qy : %f\n", *qy);
            // printf("qz : %f\n", *qz);
        }
        if(x > nBodies){
            return;
        }
        
        
        float ax = accelerations[x].ax, ay = accelerations[x].ay,
              az = accelerations[x].az;
        for (unsigned long jBody = 0; jBody < nBodies; jBody++) {

            // All forces of bodies of indexes lower than the current one have already been added to current body's
            // accel skiping.
            const float rijx = qx[jBody]- qx[x]; // 1 flop
            const float rijy = qy[jBody]- qy[x]; // 1 flop
            const float rijz = qz[jBody] - qz[x]; // 1 flop

            // compute the || rij ||² distance between body i and body j
            float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            // compute e²
            rijSquared += softSquared;

            const float pow = rijSquared * sqrtf(rijSquared); // 2 flops

            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = G * m[jBody] / pow; // 3 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            ax += ai * rijx; // 2 flops
            ay += ai * rijy; // 2 flops
            az += ai * rijz; // 2 flops

            // Adding acceleration forces to the j body as well.
        }

        accelerations[x].ax = ax;
        accelerations[x].ay = ay;
        accelerations[x].az = az;
}

void SimulationNBodyCUDA::computeOneIteration()
{
    this->initIteration();
    const float softSquared = this->soft * this->soft;
    const unsigned long nBodies = this->getBodies().getN();
    std::vector<accAoS_t<float>> h_accelerations = this->accelerations;
    dataSoA_t<float> h_bodies = this->getBodies().getDataSoA();
    const unsigned long arraySize = sizeof(float) * nBodies;

    cudaMalloc(&d_qx, arraySize);
    cudaMalloc(&d_qy, arraySize);
    cudaMalloc(&d_qz, arraySize);
    cudaMalloc(&d_m, arraySize);
    cudaMalloc(&d_accelerations, sizeof(accAoS_t<float>) * nBodies);

    cudaMemcpy(d_qx, &(h_bodies.qx), arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, &h_bodies.qy, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, &h_bodies.qz, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &h_bodies.m, arraySize, cudaMemcpyHostToDevice);

    computeBodiesAcceleration<<< blocksPerGrid,threadsPerBlock >>>(nBodies, softSquared, this->G, d_qx, d_qy, d_qz, d_m, d_accelerations);

    cudaMemcpy(this->accelerations.data(), &d_accelerations, sizeof(accAoS_t<float>) * nBodies, cudaMemcpyDeviceToHost);

    cudaFree(d_qx);
    cudaFree(d_qy);
    cudaFree(d_qz);
    cudaFree(d_m);
    cudaFree(d_accelerations); 
    
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
