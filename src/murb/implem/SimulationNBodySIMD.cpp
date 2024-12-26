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
        this->accelerations.ax.push_back(0.F);
        this->accelerations.ay.push_back(0.F);
        this->accelerations.az.push_back(0.F);
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
        for (unsigned long jBody =  iBody+1; jBody < this->getBodies().getN(); jBody+=N) {

            
            //All forces of bodies of indexes lower than the current one have already been added to current body's accel skiping.

            mipp::Reg< float> rijx = mipp::load(&d.qx[jBody]);
            mipp::Reg< float> rijy = mipp::load(&d.qy[jBody]);
            mipp::Reg< float> rijz = mipp::load(&d.qz[jBody]);

            rijx = rijx - r_iqx;
            rijy = rijy - r_iqy;
            rijz = rijz - r_iqz;
            
            // mipp::dump<float>(rijx.r);
            // mipp::dump<float>(rijy.r);
            // mipp::dump<float>(rijz.r);
            mipp::Reg<double> r_jm_1 =mipp::load(&d.m[jBody]);
            mipp::Reg<double> r_jm_2 = mipp::load(&d.m[jBody+2]);

            // compute the || rij ||² distance between body i and body j
            mipp::Reg< float>rijSquared = rijx*rijx; 
            rijSquared += rijy * rijy;
            rijSquared += rijz * rijz; // 5 flops

            mipp::Reg<double> r_pow_1 = {mipp::get(rijSquared, 0), mipp::get(rijSquared, 1)}; 
            mipp::Reg<double> r_pow_2 = {mipp::get(rijSquared, 2), mipp::get(rijSquared, 3)};
            r_pow_1 *= r_pow_1 * r_pow_1;
            r_pow_2 *= r_pow_2 * r_pow_2;
            // printf("r_pow avant revers sqrt\n");
            // mipp::dump<float>(r_pow.r);
            // printf("\n");
            r_pow_1 = mipp::rsqrt(r_pow_1);
            r_pow_2 = mipp::rsqrt(r_pow_2);
            // printf("r_pow apres reverse sqrt\n");
            // mipp::dump<double>(r_pow_1.r);
            // mipp::dump<double>(r_pow_2.r);
            // printf("\n");
            // sleep(5);
            
            //const float pow = std::pow(rijSquared + softSquared, 3.f / 2.f);// 2 flops
            
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            mipp::Reg<double> r_ai_1 = (r_jm_1 * r_pow_1) * this->G; // 3 flops
            mipp::Reg<double> r_ai_2 = (r_jm_2 * r_pow_2) * this->G; // 3 flops
            // printf("r_ai\n");
            // mipp::dump<double>(r_ai_1.r);
            // mipp::dump<double>(r_ai_2.r);
            // printf("\n");
            // sleep(5);
            

            //const float aj = this->G * d[iBody].m / pow; // 3 flops
           
            /*
              mipp::Reg<float> r_aj = (r_im * r_pow) * this->G; // 3 flops

            */

             
            // add the acceleration value into the acceleration vector: ai += || ai ||.rij

            mipp::Reg<double> r_ax_1;
            mipp::Reg<double> r_ay_1;
            mipp::Reg<double> r_az_1;
            mipp::Reg<double> r_ax_2;
            mipp::Reg<double> r_ay_2;
            mipp::Reg<double> r_az_2;
            mipp::Reg<double> rijx_1 = {mipp::get(rijx, 0), mipp::get(rijx, 1)};
            mipp::Reg<double> rijx_2 = {mipp::get(rijx, 2), mipp::get(rijx, 3)};
            mipp::Reg<double> rijy_1 = {mipp::get(rijy, 0), mipp::get(rijy, 1)};
            mipp::Reg<double> rijy_2 = {mipp::get(rijy, 2), mipp::get(rijy, 3)};
            mipp::Reg<double> rijz_1 = {mipp::get(rijz, 0), mipp::get(rijz, 1)};
            mipp::Reg<double> rijz_2 = {mipp::get(rijz, 2), mipp::get(rijz, 3)};


            r_ax_1 = (r_ai_1 * rijx_1);
            r_ax_2 = (r_ai_2 * rijx_2);
            r_ay_1 = (r_ai_1 * rijy_1);
            r_ay_2 = (r_ai_2 * rijy_2);
            r_az_1 = (r_ai_1 * rijz_1);
            r_az_2 = (r_ai_2 * rijz_2);

            this->accelerations.ax[iBody] += mipp::hadd<double>(r_ax_1) + mipp::hadd<double>(r_ax_2);
            this->accelerations.ay[iBody] += mipp::hadd<double>(r_ay_1) + mipp::hadd<double>(r_ay_2);
            this->accelerations.az[iBody] += mipp::hadd<double>(r_az_1) + mipp::hadd<double>(r_az_2);
            

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
