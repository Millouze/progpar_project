#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <driver_types.h>

#include "SimulationNBodyCUDA.hpp"
#include "core/Bodies.hpp"

static dim3 blocksPerGrid = {1};
static dim3 threadsPerBlock = {1024};
static float *qx, *qy, *qz, *m;
static accAoS_t<float> *d_accelerations;

// SimulationNBodyCUDA::SimulationNBodyCUDA(const unsigned long nBodies, const std::string &scheme, const float soft,
//                                            const unsigned long randInit)
//     : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
// {
//     this->flopsPerIte = 27.f * (((float)this->getBodies().getN()+1) * (float)this->getBodies().getN())/2;
//     //this->accelerations.resize(this->getBodies().getN());
//     this->accelerations.ax.resize(this->getBodies().getN() + this->getBodies().getPadding());
//     this->accelerations.ay.resize(this->getBodies().getN() + this->getBodies().getPadding());
//     this->accelerations.az.resize(this->getBodies().getN() + this->getBodies().getPadding());
// }

// void SimulationNBodyCUDA::initIteration()
// {
//     for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
//         this->accelerations.ax.push_back(0.f);
//         this->accelerations.ay.push_back(0.f);
//         this->accelerations.az.push_back(0.f);
//     }
// }

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
            printf("qx : %f\n", *qx);
            printf("qy : %f\n", *qy);
            printf("qz : %f\n", *qz);
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

            const float pow = rsqrtf(rijSquared); // 2 flops

            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = G * m[jBody] * (pow * pow * pow); // 3 flops

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
    unsigned long arraySize = sizeof(float) * nBodies;


    cudaMalloc(&qx, arraySize);
    cudaMalloc(&qy, arraySize);
    cudaMalloc(&qz, arraySize);
    cudaMalloc(&m, arraySize);
    // cudaMalloc(&ax, arraySize);
    // cudaMalloc(&ay, arraySize);
    // cudaMalloc(&az, arraySize);
    cudaMalloc(&d_accelerations, sizeof(struct accAoS_t<float>) * nBodies);

    // memset(ax, 0, arraySize);
    // memset(ay, 0, arraySize);
    // memset(az, 0, arraySize);

    cudaMemcpy(qx, &h_bodies.qx, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(qy, &h_bodies.qy, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(qz, &h_bodies.qz, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(m, &h_bodies.m, arraySize, cudaMemcpyHostToDevice);
    // cudaMemcpy(ax, &h_accelerations.ax, arraySize, cudaMemcpyHostToDevice);
    // cudaMemcpy(ay, &h_accelerations.ay, arraySize, cudaMemcpyHostToDevice);
    // cudaMemcpy(az, &h_accelerations.az, arraySize, cudaMemcpyHostToDevice);

    // memset(ax, 0, arraySize);
    // memset(ay, 0, arraySize);
    // memset(az, 0, arraySize);



    computeBodiesAcceleration<<< blocksPerGrid,threadsPerBlock >>>(nBodies, softSquared, this->G, qx, qy, qz, m, d_accelerations);

    // cudaMemcpy(&this->accelerations.ax, ax, arraySize, cudaMemcpyDeviceToHost);
    // cudaMemcpy(&this->accelerations.ay, ay, arraySize, cudaMemcpyDeviceToHost);
    // cudaMemcpy(&this->accelerations.az, az, arraySize, cudaMemcpyDeviceToHost);
    //
    cudaMemcpy(this->accelerations.data(), d_accelerations, sizeof(struct accAoS_t<float>) * nBodies, cudaMemcpyDeviceToHost);

    cudaFree(qx);
    cudaFree(qy);
    cudaFree(qz);
    cudaFree(m);
    // cudaFree(ax);
    // cudaFree(ay);
    // cudaFree(az);
    
    cudaFree(d_accelerations); 
    
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
