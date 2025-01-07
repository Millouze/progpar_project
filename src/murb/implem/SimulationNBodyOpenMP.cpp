
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <omp.h>

#include "SimulationNBodyOpenMP.hpp"

SimulationNBodyOpenMP::SimulationNBodyOpenMP(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 27.f * (((float)this->getBodies().getN()+1) * (float)this->getBodies().getN())/2;
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyOpenMP::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}


//We delete unecessary double calculation of forces by using the reciprocity of gravitational pull.

void SimulationNBodyOpenMP::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    const float softSquared = this->soft *  this->soft;// 1 flops
    // flops = n² * 20
    #pragma omp parallel for schedule(dynamic, 20)
    
        for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        // flops = n * 20
        //#pragma omp parallel num_threads(4)
        //{
            float ax = this->accelerations[iBody].ax, ay = this->accelerations[iBody].ay, az = this->accelerations[iBody].az;
<<<<<<< HEAD
            //#pragma omp for schedule(static, 10) nowait 
            
            for (unsigned long jBody = iBody+1; jBody < this->getBodies().getN(); jBody++) {
=======
            #pragma omp for schedule(static, 16) nowait
                for (unsigned long jBody = iBody+1; jBody < this->getBodies().getN(); jBody++) {
>>>>>>> 3e0fd79e7e91c6acd4af5876b79aa7382d4a10b2

                //All forces of bodies of indexes lower than the current one have already been added to current body's accel skiping.
                const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
                const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
                const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

<<<<<<< HEAD
            // compute the || rij ||² distance between body i and body j
            float rijSquared = rijx*rijx + rijy * rijy + rijz * rijz; // 5 flops
            rijSquared += softSquared;
            // compute e²
            
            
            float pow =  rijSquared * std::sqrt(rijSquared);
=======
                // compute the || rij ||² distance between body i and body j
                const float rijSquared = rijx*rijx + rijy * rijy + rijz * rijz; // 5 flops
                // compute e²
            
            
                const float pow = std::pow(rijSquared + softSquared, 3.f / 2.f);// 2 flops
>>>>>>> 3e0fd79e7e91c6acd4af5876b79aa7382d4a10b2
            
                // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
                const float ai = this->G * d[jBody].m / pow; // 3 flops
            

<<<<<<< HEAD
            const float aj = this->G * d[iBody].m / pow; // 3 flops
            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            ax += ai * rijx; // 2 flops
            ay += ai * rijy; // 2 flops
            az += ai * rijz; // 2 flops

            //Adding acceleration forces to the j body as well.
            this->accelerations[jBody].ax -= aj * rijx; // 2 flops
            this->accelerations[jBody].ay -= aj * rijy; // 2 flops
            this->accelerations[jBody].az -= aj * rijz; // 2 flops
            }
            
=======
                const float aj = this->G * d[iBody].m / pow; // 3 flops
                // add the acceleration value into the acceleration vector: ai += || ai ||.rij
                ax += ai * rijx; // 2 flops
                ay += ai * rijy; // 2 flops
                az += ai * rijz; // 2 flops
>>>>>>> 3e0fd79e7e91c6acd4af5876b79aa7382d4a10b2

                //Adding acceleration forces to the j body as well.
                this->accelerations[jBody].ax -= aj * rijx; // 2 flops
                this->accelerations[jBody].ay -= aj * rijy; // 2 flops
                this->accelerations[jBody].az -= aj * rijz; // 2 flops
                }
            
            
            this->accelerations[iBody].ax = ax;
            this->accelerations[iBody].ay = ay;
            this->accelerations[iBody].az = az;
        }
<<<<<<< HEAD
    
=======
>>>>>>> 3e0fd79e7e91c6acd4af5876b79aa7382d4a10b2
        
        
    
}

void SimulationNBodyOpenMP::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
