#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda.h>
#include <math.h>
#include <omp.h>

#include "SimulationNBodyHybrid.hpp"

#define CPU_PERCENTAGE 0.10

__constant__ float softSquared;
__constant__ float gravity;

SimulationNBodyHybrid::SimulationNBodyHybrid(const unsigned long nBodies, const std::string &scheme, const float soft,
                                             const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    if((int)(nBodies*(1-CPU_PERCENTAGE))%32 != 0) {
        NforWarp = (((int)(nBodies*(1-CPU_PERCENTAGE))/32)+1)*32;
    } else {
        NforWarp = nBodies*(1-CPU_PERCENTAGE);
    }

    NforProcs = this->getBodies().getN() - NforWarp;

        
    const float hsoft = this->soft * this->soft;
    const unsigned long arraySize = sizeof(float) * nBodies;
    const unsigned long accSize = sizeof(struct accAoS_t<float>) * nBodies;
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
    cudaMemcpyToSymbol(softSquared, &hsoft, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gravity, &this->G, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMalloc(&d_qx, arraySize);
    cudaMalloc(&d_qy, arraySize);
    cudaMalloc(&d_qz, arraySize);
    cudaMalloc(&d_m, arraySize);
    cudaMalloc(&d_accelerations, accSize);
}

SimulationNBodyHybrid::~SimulationNBodyHybrid() {
      cudaFree(d_qx);
      cudaFree(d_qy);
      cudaFree(d_qz);
      cudaFree(d_m);
      cudaFree(d_accelerations);
    }

void SimulationNBodyHybrid::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

static __global__ void computeBodiesAccelerationGPU(const unsigned long nBodies,
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

void SimulationNBodyHybrid::computeBodiesAccelerationCPU()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    const float softSquared = this->soft * this->soft; // 1 flops
                                                       // flops = n² * 20
    #pragma omp parallel for schedule(dynamic, 20) \
        firstprivate(d)
    for (unsigned long iBody = NforProcs; iBody < this->getBodies().getN(); iBody++) {
        // flops = n * 20
        float ax = this->accelerations[iBody].ax, ay = this->accelerations[iBody].ay,
              az = this->accelerations[iBody].az;
            for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {

            // All forces of bodies of indexes lower than the current one have already been added to current body's
            // accel skiping.
            const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            // compute e²

            const float pow = rijSquared * sqrtf(rijSquared); // 2 flops

            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = this->G * d[jBody].m / pow; // 3 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            ax += ai * rijx; // 2 flops
            ay += ai * rijy; // 2 flops
            az += ai * rijz; // 2 flops

        }

        this->accelerations[iBody].ax = ax;
        this->accelerations[iBody].ay = ay;
        this->accelerations[iBody].az = az;
    }
}

void SimulationNBodyHybrid::computeOneIteration()
{
    this->initIteration();
    //const float softSquared = this->soft * this->soft;
    const unsigned long nBodies = this->getBodies().getN();
    dataSoA_t<float> h_bodies = this->getBodies().getDataSoA();
    const unsigned long arraySize = sizeof(float) * (nBodies);
    const unsigned long accSize = sizeof(struct accAoS_t<float>) * (nBodies);
 
    cudaMemcpy(d_qx, &(h_bodies.qx[0]), arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, &(h_bodies.qy[0]), arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, &(h_bodies.qz[0]), arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &(h_bodies.m[0]), arraySize, cudaMemcpyHostToDevice);
   
    dim3 threadsPerBlock = {512};
    dim3 blocksPerGrid = {(((nBodies-NforProcs))+threadsPerBlock.x -1)/threadsPerBlock.x};
    

    computeBodiesAccelerationGPU<<< blocksPerGrid,threadsPerBlock >>>(nBodies, d_qx, d_qy, d_qz, d_m, d_accelerations);
    computeBodiesAccelerationCPU();


    cudaMemcpy(this->accelerations.data(), d_accelerations, accSize, cudaMemcpyDeviceToHost);
  
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
