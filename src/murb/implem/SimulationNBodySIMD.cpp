#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <ostream>
#include <string>

#include "../../../lib/MIPP/src/mipp.h"

#include "SimulationNBodySIMD.hpp"

SimulationNBodySIMD::SimulationNBodySIMD(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 27.f * (((float)this->getBodies().getN()+1) * (float)this->getBodies().getN())/2;
    //this->accelerations.resize(this->getBodies().getN());
    this->accelerations.ax.resize(this->getBodies().getN() + this->getBodies().getPadding());
    this->accelerations.ay.resize(this->getBodies().getN() + this->getBodies().getPadding());
    this->accelerations.az.resize(this->getBodies().getN() + this->getBodies().getPadding());
}

void SimulationNBodySIMD::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations.ax.push_back(0.f);
        this->accelerations.ay.push_back(0.f);
        this->accelerations.az.push_back(0.f);
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


void print_reg(mipp::Reg<float> reg) {
    constexpr int N = mipp::N<float>();
    float mem[N];
    reg.store(mem);
    printf("{");
    for (int i=0;i<N;i++) {
        printf(", %f ",mem[i]);
    }
    printf("\b}\n");
}

//We delete unecessary double calculation of forces by using the reciprocity of gravitational pull.

void SimulationNBodySIMD::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();


    const float softSquared = this->soft *  this->soft;// 1 flops

    //SIMD Internal loop pitch
    constexpr int N = mipp::N<float>();
    mipp::Reg<float> grav = this->G;
    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        // flops = n * 20

        mipp::Reg<float> r_iqx = d.qx[iBody];
        // if (!iBody) {
        //     print_reg(r_iqx);
        // }
        mipp::Reg<float> r_iqy = d.qy[iBody];
        mipp::Reg<float> r_iqz = d.qz[iBody];
        mipp::Reg<float> r_im = d.m[iBody];
        // if (!iBody) {
        //     print_reg(r_im);
        // }
        for (unsigned long jBody =  0; jBody < this->getBodies().getN(); jBody+=N) {

            
            //All forces of bodies of indexes lower than the current one have already been added to current body's accel skiping.
            // const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            // const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            // const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            mipp::Reg< float> rijx = mipp::load(&d.qx[jBody]);
            mipp::Reg< float> rijy = mipp::load(&d.qy[jBody]);
            mipp::Reg< float> rijz = mipp::load(&d.qz[jBody]);

            rijx = rijx - r_iqx;
            rijy = rijy - r_iqy;
            rijz = rijz - r_iqz;
            
            mipp::Reg<float> r_jm = mipp::load(&d.m[jBody]);

            // print_reg(r_jm);
            // compute the || rij ||² distance between body i and body j
            mipp::Reg< float>rijSquared = rijx*rijx; 
            rijSquared += rijy * rijy;
            rijSquared += rijz * rijz; // 5 flops
            // compute e²

            mipp::Reg<float> r_pow = rijSquared+softSquared;
            r_pow *= r_pow*r_pow;
            r_pow = mipp::rsqrt(r_pow);
            
            //const float pow = std::pow(rijSquared + softSquared, 3.f / 2.f);// 2 flops
            
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            mipp::Reg<float> r_ai = (r_jm * r_pow) * grav; // 3 flops
            

            //const float aj = this->G * d[iBody].m / pow; // 3 flops
           
            /*
              mipp::Reg<float> r_aj = (r_im * r_pow) * this->G; // 3 flops

            */

             
            // add the acceleration value into the acceleration vector: ai += || ai ||.rij

            mipp::Reg<float> r_ax;
            mipp::Reg<float> r_ay;
            mipp::Reg<float> r_az;

            r_ax = (r_ai * rijx);
            r_ay = (r_ai * rijy);
            r_az = (r_ai * rijz);

            this->accelerations.ax[iBody] += mipp::hadd<float>(r_ax);
            this->accelerations.ay[iBody] += mipp::hadd<float>(r_ay);
            this->accelerations.az[iBody] += mipp::hadd<float>(r_az);
            
            // this->accelerations.ax[iBody] += ai * rijx; // 2 flops
            // this->accelerations[iBody].ay += ai * rijy; // 2 flops
            // this->accelerations[iBody].az += ai * rijz; // 2 flops

            //Adding acceleration forces to the j body as well.
            
            
            /*
            r_ax = mipp::load(&this->accelerations.ax[jBody]);
            r_ay = mipp::load(&this->accelerations.ay[jBody]);
            r_az = mipp::load(&this->accelerations.az[jBody]);
            
            r_ax = r_ax - (r_aj * rijx);
            r_ay = r_ax - (r_aj * rijy);
            r_az = r_ax - (r_aj * rijz);
            // this->accelerations[jBody].ax -= aj * rijx; // 2 flops
            // this->accelerations[jBody].ay -= aj * rijy; // 2 flops
            // this->accelerations[jBody].az -= aj * rijz; // 2 flops
            mipp::store(&this->accelerations.ax[jBody],r_ax );
            mipp::store(&this->accelerations.ay[jBody],r_ay );
            mipp::store(&this->accelerations.az[jBody],r_az ); 
            
            */
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
