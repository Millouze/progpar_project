#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyLessComplex.hpp"

SimulationNBodyLessComplex::SimulationNBodyLessComplex(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 27.f * (((float)this->getBodies().getN()+1) * (float)this->getBodies().getN())/2;
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyLessComplex::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}


//We delete unecessary double calculation of forces by using the reciprocity of gravitational pull.

void SimulationNBodyLessComplex::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    const float softSquared = this->soft *  this->soft;// 1 flops
    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        // flops = n * 20
        float ax = this->accelerations[iBody].ax, ay = this->accelerations[iBody].ay, az = this->accelerations[iBody].az;
        for (unsigned long jBody = iBody+1; jBody < this->getBodies().getN(); jBody++) {

            //All forces of bodies of indexes lower than the current one have already been added to current body's accel skiping.
            const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = rijx*rijx + rijy * rijy + rijz * rijz; // 5 flops
            // compute e²
            
            
            const float pow = std::pow(rijSquared + softSquared, 3.f / 2.f);// 2 flops
            
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = this->G * d[jBody].m / pow; // 3 flops
            

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

        this->accelerations[iBody].ax = ax;
        this->accelerations[iBody].ay = ay;
        this->accelerations[iBody].az = az;
    }
}

void SimulationNBodyLessComplex::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
