#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "SimulationNBodyAoS.hpp"
#include "core/Bodies.hpp"

static dim3 blocksPerGrid = {60};
static dim3 threadsPerBlock = {1024};
dataAoS_t<float> *d_bodies;
accAoS_t<float> *d_accelerations;

SimulationNBodyAoS::SimulationNBodyAoS(const unsigned long nBodies, const std::string &scheme, const float soft,
                                         const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyAoS::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

__global__ void computeBodiesAcceleration(const unsigned long nBodies, const float softSquared, const float G,
                                          dataAoS_t<float> *d, accAoS_t<float> *accelerations)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;

        if(x > nBodies){
            return;
        }
        
        float ax = accelerations[x].ax, ay = accelerations[x].ay,
              az = accelerations[x].az;
        for (unsigned long jBody = 0; jBody < nBodies; jBody++) {

            // All forces of bodies of indexes lower than the current one have already been added to current body's
            // accel skiping.
            const float rijx = d[jBody].qx - d[x].qx; // 1 flop
            const float rijy = d[jBody].qy - d[x].qy; // 1 flop
            const float rijz = d[jBody].qz - d[x].qz; // 1 flop

            // compute the || rij ||² distance between body i and body j
            float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            // compute e²
            rijSquared += softSquared;

            const float pow = rsqrtf(rijSquared); // 2 flops

            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = G * d[jBody].m * (pow * pow * pow); // 3 flops

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

void SimulationNBodyAoS::computeOneIteration()
{
    this->initIteration();
    const float softSquared = this->soft * this->soft;
    const unsigned long nBodies = this->getBodies().getN();
    const std::vector<dataAoS_t<float>> h_bodies = this->getBodies().getDataAoS();
    std::vector<accAoS_t<float>> h_accelerations = this->accelerations;

    cudaMalloc(&d_bodies, sizeof(struct dataAoS_t<float>) * nBodies);
    cudaMalloc(&d_accelerations, sizeof(struct accAoS_t<float>) * nBodies);

    cudaMemcpy(d_bodies, h_bodies.data(), sizeof(struct dataAoS_t<float>) * nBodies, cudaMemcpyHostToDevice);

    computeBodiesAcceleration<<< blocksPerGrid,threadsPerBlock >>>(this->getBodies().getN(), softSquared, this->G, d_bodies, d_accelerations);

    
    cudaGetLastError(); 
    cudaDeviceSynchronize();
    
    cudaMemcpy(this->accelerations.data(), d_accelerations, sizeof(struct accAoS_t<float>) * nBodies, cudaMemcpyDeviceToHost);

    cudaFree(d_bodies);
    cudaFree(d_accelerations);
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
} 
