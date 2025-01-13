#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <driver_types.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "SimulationNBodyCUDA_opti.hpp"
#include "core/Bodies.hpp"

// static dim3 blocksPerGrid = {1};
// static dim3 threadsPerBlock = {1024};
// static float *d_qx, *d_qy, *d_qz, *d_m;
// static accAoS_t<float> *d_accelerations;

//static dim3 blocksPerGrid = {60} ;
//static dim3 threadsPerBlock = {1024};

__constant__ float softSquared;
__constant__ float gravity;


SimulationNBodyCUDA_opti::SimulationNBodyCUDA_opti(const unsigned long nBodies, const std::string &scheme, const float soft,
                                         const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{

    const float hsoft = this->soft * this->soft;
    
    cudaMemcpyToSymbol(softSquared, &hsoft, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gravity, &this->G, sizeof(float), 0, cudaMemcpyHostToDevice);

    const unsigned long arraySize = sizeof(float) * nBodies;
    const unsigned long accSize = sizeof(struct accAoS_t<float>) * nBodies;
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
    cudaMalloc(&d_qx, arraySize);
    cudaMalloc(&d_qy, arraySize);
    cudaMalloc(&d_qz, arraySize);
    cudaMalloc(&d_m, arraySize);
    cudaMalloc(&d_accelerations, accSize);
    //blocksPerGrid = {(unsigned int)this->getBodies().getN() / 1024};
    //printf("blockspergrid %d\n",blocksPerGrid.x);
}

SimulationNBodyCUDA_opti:: ~SimulationNBodyCUDA_opti() {
      cudaFree(d_qx);
      cudaFree(d_qy);
      cudaFree(d_qz);
      cudaFree(d_m);
      cudaFree(d_accelerations);
    }

void SimulationNBodyCUDA_opti::initIteration()
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
        int x = (blockDim.x * blockIdx.x + threadIdx.x)*2;
        int x_2 = (blockDim.x * blockIdx.x + threadIdx.x)*2 +1;
        if(x > nBodies){
            return;
        }
        //printf("blockspergrid %d\n",blocksPerGrid.x);

        
        float ax = accelerations[x].ax, ay = accelerations[x].ay,
              az = accelerations[x].az;
        float ax_2 = accelerations[x_2].ax, ay_2 = accelerations[x_2].ay,
              az_2 = accelerations[x_2].az;
        for (unsigned long jBody = 0; jBody < nBodies; jBody++) {

            // All forces of bodies of indexes lower than the current one have already been added to current body's
            // accel skiping.
            const float rijx = qx[jBody]- qx[x]; // 1 flop
            const float rijy = qy[jBody]- qy[x]; // 1 flop
            const float rijz = qz[jBody] - qz[x]; // 1 flop

            const float rijx_2 = qx[jBody]- qx[x_2]; // 1 flop
            const float rijy_2 = qy[jBody]- qy[x_2]; // 1 flop
            const float rijz_2 = qz[jBody] - qz[x_2]; // 1 flop

            // compute the || rij ||² distance between body i and body j
            float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            // compute e²
            rijSquared += softSquared;

            float rijSquared_2 = rijx_2 * rijx_2 + rijy_2 * rijy_2 + rijz_2 * rijz_2; // 5 flops
            // compute e²
            rijSquared_2 += softSquared;


            const float pow = rsqrtf(rijSquared); // 2 flops

             const float pow_2 = rsqrtf(rijSquared_2); // 2 flops

            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = m[jBody] * (pow * pow * pow); // 3 flops

            const float ai_2 = m[jBody] * (pow_2 * pow_2 * pow_2); // 3 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            ax += ai * rijx; // 2 flops
            ay += ai * rijy; // 2 flops
            az += ai * rijz; // 2 flops

            ax_2 += ai_2 * rijx_2; // 2 flops
            ay_2 += ai_2 * rijy_2; // 2 flops
            az_2 += ai_2 * rijz_2; // 2 flops
            
            // accelerations[x].ax += ai *rijx;            
            // accelerations[x].ay += ai * rijy;            
            // accelerations[x].az += ai * rijz;            

            // Adding acceleration forces to the j body as well.
        }

        accelerations[x].ax = ax * gravity;
        accelerations[x].ay = ay * gravity;
        accelerations[x].az = az * gravity;

        accelerations[x_2].ax = ax_2 * gravity;
        accelerations[x_2].ay = ay_2 * gravity;
        accelerations[x_2].az = az_2 * gravity;
}

void SimulationNBodyCUDA_opti::computeOneIteration()
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
    dim3 blocksPerGrid = {(((nBodies+1) /2)+threadsPerBlock.x -1)/threadsPerBlock.x};
    

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
