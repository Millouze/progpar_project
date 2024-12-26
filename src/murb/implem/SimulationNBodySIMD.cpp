#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "../../../lib/MIPP/src/mipp.h"

#include "SimulationNBodySIMD.hpp"

SimulationNBodySIMD::SimulationNBodySIMD(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 27.f * (((float)this->getBodies().getN()+1) * (float)this->getBodies().getN())/2;
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodySIMD::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}


//Quake's fast inverse square root but now MIPPed
mipp::Reg<float> Q_rsqrt( mipp::Reg<float> number )
{
	long i;
	mipp::Reg<float> x2, y;
	mipp::Reg<float> threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;						// evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
//	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}

//We delete unecessary double calculation of forces by using the reciprocity of gravitational pull.

void SimulationNBodySIMD::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();


    const float softSquared = this->soft *  this->soft;// 1 flops

    //SIMD Internal loop pitch
    constexpr int N = mipp::N<float>();
    
    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        // flops = n * 20

        mipp::Reg<float> r_iqx = d.qx[iBody];
        mipp::Reg<float> r_iqy = d.qy[iBody];
        mipp::Reg<float> r_iqz = d.qz[iBody];
        mipp::Reg<float> r_im = d.m[iBody];
        
        for (unsigned long jBody = iBody+1; jBody < this->getBodies().getN(); jBody+=N) {
            
            //All forces of bodies of indexes lower than the current one have already been added to current body's accel skiping.
            // const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            // const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            // const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            mipp::Reg< float> rijx = mipp::load(&d.qx[jBody]);
            mipp::Reg< float> rijy = mipp::load(&d.qy[jBody]);
            mipp::Reg< float> rijz = mipp::load(&d.qz[jBody]);
            mipp::Reg<float> r_jm = d.m[jBody];

            // compute the || rij ||² distance between body i and body j
            mipp::Reg< float>rijSquared = rijx*rijx + rijy * rijy + rijz * rijz; // 5 flops
            // compute e²

            mipp::Reg<float> pow = Q_rsqrt(rijSquared+softSquared);
            pow *= pow*pow;
            
            //const float pow = std::pow(rijSquared + softSquared, 3.f / 2.f);// 2 flops
            
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            mipp::Reg<float> ai = (r_jm * pow) * this->G; // 3 flops
            

            //const float aj = this->G * d[iBody].m / pow; // 3 flops
            mipp::Reg<float> aj = (r_im * pow) * this->G; // 3 flops
            // add the acceleration value into the acceleration vector: ai += || ai ||.rij

            // FIXME need to figure out how to store the acceleration from the SIMD register
            // to the NBody Structure.
            this->accelerations[iBody].ax += ai * rijx; // 2 flops
            this->accelerations[iBody].ay += ai * rijy; // 2 flops
            this->accelerations[iBody].az += ai * rijz; // 2 flops

            //Adding acceleration forces to the j body as well.
            this->accelerations[jBody].ax -= aj * rijx; // 2 flops
            this->accelerations[jBody].ay -= aj * rijy; // 2 flops
            this->accelerations[jBody].az -= aj * rijz; // 2 flops
        }
    }
}

void SimulationNBodySIMD::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
