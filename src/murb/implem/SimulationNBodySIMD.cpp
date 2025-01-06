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
        printf("%f, ",mem[i]);
    }
    printf("\b}\n");
}


inline mipp::Reg<float> heron_sqrt(const mipp::Reg<float> a)
{
    mipp::Reg<float> point_five = mipp::set1<float>(0.5f);
    mipp::Reg<float> res = mipp::sqrt(a);
    res = mipp::fmadd(point_five ,res, point_five * (a / res));
    return res;
}

//We delete unecessary double calculation of forces by using the reciprocity of gravitational pull.

void SimulationNBodySIMD::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();


    const float softSquared = this->soft *  this->soft;// 1 flops

    //SIMD Internal loop pitch
    constexpr int N = mipp::N<float>();
    unsigned long loop_tail = (this->getBodies().getN()/N)*N;
    // printf("loop tail = %lu\n",loop_tail);
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
            mipp::Reg<float> r_ax = mipp::set0<float>();
            mipp::Reg<float> r_ay = mipp::set0<float>();
            mipp::Reg<float> r_az = mipp::set0<float>();
        for (unsigned long jBody = 0; jBody < loop_tail; jBody+=N) {

            
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
            r_pow = r_pow*mipp::sqrt(r_pow);
            // r_pow = mipp::rsqrt(r_pow);
            
            //const float pow = std::pow(rijSquared + softSquared, 3.f / 2.f);// 2 flops
            
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            mipp::Reg<float> r_ai = grav*r_jm / r_pow; // 3 flops
            

            //const float aj = this->G * d[iBody].m / pow; // 3 flops
           
            /*
              mipp::Reg<float> r_aj = (r_im * r_pow) * this->G; // 3 flops

            */

             
            // add the acceleration value into the acceleration vector: ai += || ai ||.rij

            

            r_ax += (r_ai * rijx);
            r_ay += (r_ai * rijy);
            r_az += (r_ai * rijz);

            
            
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
            this->accelerations.ax[iBody] += mipp::hadd<float>(r_ax);
            this->accelerations.ay[iBody] += mipp::hadd<float>(r_ay);
            this->accelerations.az[iBody] += mipp::hadd<float>(r_az);

        for(unsigned long jBody = loop_tail; jBody < this->getBodies().getN(); jBody++){
            
            const float rijx = d.qx[jBody] - d.qx[iBody]; // 1 flop
            const float rijy = d.qy[jBody] - d.qy[iBody]; // 1 flop
            const float rijz = d.qz[jBody] - d.qz[iBody]; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = rijx * rijx + rijy * rijy + rijz *rijz; // 5 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = this->G * d.m[jBody] / std::pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            this->accelerations.ax[iBody] += ai * rijx; // 2 flops
            this->accelerations.ay[iBody] += ai * rijy; // 2 flops
            this->accelerations.az[iBody] += ai * rijz; // 2 flops
        }
    }
}

void SimulationNBodySIMD::computeBodiesAccelerationV2()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();


    const float softSquared = this->soft *  this->soft;// 1 flops

    //SIMD Internal loop pitch
    constexpr int N = mipp::N<float>();
    unsigned long loop_tail = (this->getBodies().getN()/N)*N;
    // printf("loop tail = %lu\n",loop_tail);
    mipp::Reg<float> grav = this->G;
    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        // flops = n * 20

        mipp::Reg<float> r_iqx = d.qx[iBody];
        mipp::Reg<float> r_iqy = d.qy[iBody];
        mipp::Reg<float> r_iqz = d.qz[iBody];
        mipp::Reg<float> r_im = d.m[iBody];

        float iqx = d.qx[iBody];
        printf("\n\n\n\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
        print_reg(r_iqx);
        printf("%f\n",iqx);
        float iqy = d.qy[iBody];
        float iqz = d.qz[iBody];
        
        // if (!iBody) {
        //     print_reg(r_im);
        // }
            mipp::Reg<float> r_ax = mipp::set0<float>();
            mipp::Reg<float> r_ay = mipp::set0<float>();
            mipp::Reg<float> r_az = mipp::set0<float>();
        for (unsigned long jBody = 0; jBody < loop_tail; jBody+=N) {

            
            //All forces of bodies of indexes lower than the current one have already been added to current body's accel skiping.
            // const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            // const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            // const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            mipp::Reg< float> rijx = mipp::load(&d.qx[jBody]);
            mipp::Reg< float> rijy = mipp::load(&d.qy[jBody]);
            mipp::Reg< float> rijz = mipp::load(&d.qz[jBody]);

            float rijx_0 = d.qx[jBody];
            float rijx_1 = d.qx[jBody+1];
            float rijx_2 = d.qx[jBody+2];
            float rijx_3 = d.qx[jBody+3];
            float rijx_4 = d.qx[jBody+4];           
            float rijx_5 = d.qx[jBody+5];
            float rijx_6 = d.qx[jBody+6];           
            float rijx_7 = d.qx[jBody+7];           

            float rijy_0 = d.qy[jBody];
            float rijy_1 = d.qy[jBody+1];
            float rijy_2 = d.qy[jBody+2];
            float rijy_3 = d.qy[jBody+3];
            float rijy_4 = d.qy[jBody+4];           
            float rijy_5 = d.qy[jBody+5];
            float rijy_6 = d.qy[jBody+6];           
            float rijy_7 = d.qy[jBody+7];           

            float rijz_0 = d.qz[jBody];
            float rijz_1 = d.qz[jBody+1];
            float rijz_2 = d.qz[jBody+2];
            float rijz_3 = d.qz[jBody+3];
            float rijz_4 = d.qz[jBody+4];           
            float rijz_5 = d.qz[jBody+5];
            float rijz_6 = d.qz[jBody+6];           
            float rijz_7 = d.qz[jBody+7];           


            
            // printf("\n\n\nChargement des valeurs des jBodies sur x\n");
            // printf("{%f, %f, %f, %f, %f, %f, %f, %f}\n",rijx_0,rijx_1,rijx_2,rijx_3,rijx_4,rijx_5,rijx_6,rijx_7);
            // print_reg(rijx);
            
            // printf("Chargement des valeurs des jBodies sur y\n");
            // printf("{%f, %f, %f, %f, %f, %f, %f, %f}\n",rijy_0,rijy_1,rijy_2,rijy_3,rijy_4,rijy_5,rijy_6,rijy_7);
            // print_reg(rijy);
            
            // printf("Chargement des valeurs des jBodies sur z\n");
            // printf("{%f, %f, %f, %f, %f, %f, %f, %f}\n",rijz_0,rijz_1,rijz_2,rijz_3,rijz_4,rijz_5,rijz_6,rijz_7);
            // print_reg(rijz);

            //SIMD
            rijx = rijx - r_iqx;
            rijy = rijy - r_iqy;
            rijz = rijz - r_iqz;

            //PAS SIMD
            rijx_0 -=iqx;
            rijx_1 -=iqx;
            rijx_2 -=iqx;
            rijx_3 -=iqx;
            rijx_4 -=iqx;
            rijx_5 -=iqx;
            rijx_6 -=iqx;
            rijx_7 -=iqx;
            
            rijy_0 -=iqy;
            rijy_1 -=iqy;
            rijy_2 -=iqy;
            rijy_3 -=iqy;
            rijy_4 -=iqy;
            rijy_5 -=iqy;
            rijy_6 -=iqy;
            rijy_7 -=iqy;

            
            rijz_0 -=iqz;
            rijz_1 -=iqz;
            rijz_2 -=iqz;
            rijz_3 -=iqz;
            rijz_4 -=iqz;
            rijz_5 -=iqz;
            rijz_6 -=iqz;
            rijz_7 -=iqz;
            
            // printf("\n\n Soustraction avec le iBody des valeurs des jBodies sur x\n");
            // printf("{%f, %f, %f, %f, %f, %f, %f, %f}\n",rijx_0,rijx_1,rijx_2,rijx_3,rijx_4,rijx_5,rijx_6,rijx_7);
            // print_reg(rijx);
            
            // printf("Soustraction avec le ibod des jBodies sur y\n");
            // printf("{%f, %f, %f, %f, %f, %f, %f, %f}\n",rijy_0,rijy_1,rijy_2,rijy_3,rijy_4,rijy_5,rijy_6,rijy_7);
            // print_reg(rijy);
            
            // printf("Soustraction avec le ibod des jBodies sur z\n");
            // printf("{%f, %f, %f, %f, %f, %f, %f, %f}\n",rijz_0,rijz_1,rijz_2,rijz_3,rijz_4,rijz_5,rijz_6,rijz_7);
            // print_reg(rijz);

            
            mipp::Reg<float> r_jm = mipp::load(&d.m[jBody]);

            float jm_0 = d.m[jBody];
            float jm_1 = d.m[jBody+1];
            float jm_2 = d.m[jBody+2];
            float jm_3 = d.m[jBody+3];
            float jm_4 = d.m[jBody+4];           
            float jm_5 = d.m[jBody+5];
            float jm_6 = d.m[jBody+6];           
            float jm_7 = d.m[jBody+7];           

            // print_reg(r_jm);
            // compute the || rij ||² distance between body i and body j
            mipp::Reg< float>rijSquared = rijx*rijx; 
            rijSquared += rijy * rijy;
            rijSquared += rijz * rijz; // 5 flops
            // compute e²

            float rijSquared_0 = rijx_0*rijx_0 + rijy_0*rijy_0 + rijz_0*rijz_0; 
            float rijSquared_1 = rijx_1*rijx_1 + rijy_1*rijy_1 + rijz_1*rijz_1; 
            float rijSquared_2 = rijx_2*rijx_2 + rijy_2*rijy_2 + rijz_2*rijz_2; 
            float rijSquared_3 = rijx_3*rijx_3 + rijy_3*rijy_3 + rijz_3*rijz_3; 
            float rijSquared_4 = rijx_4*rijx_4 + rijy_4*rijy_4 + rijz_4*rijz_4; 
            float rijSquared_5 = rijx_5*rijx_5 + rijy_5*rijy_5 + rijz_5*rijz_5; 
            float rijSquared_6 = rijx_6*rijx_6 + rijy_6*rijy_6 + rijz_6*rijz_6; 
            float rijSquared_7 = rijx_7*rijx_7 + rijy_7*rijy_7 + rijz_7*rijz_7; 
            

            // printf("\n\n rijsquared\n");
            // printf("{%f, %f, %f, %f, %f, %f, %f, %f}\n",rijSquared_0,rijSquared_1,rijSquared_2,rijSquared_3,rijSquared_4,rijSquared_5,rijSquared_6,rijSquared_7);
            // print_reg(rijSquared);
            

            mipp::Reg<float> r_pow = rijSquared+softSquared;
            r_pow = r_pow*mipp::sqrt(r_pow);
            // r_pow = mipp::rsqrt(r_pow);

            float pow_0 = rijSquared_0 + softSquared; 
            float pow_1 = rijSquared_1 + softSquared; 
            float pow_2 = rijSquared_2 + softSquared; 
            float pow_3 = rijSquared_3 + softSquared; 
            float pow_4 = rijSquared_4 + softSquared; 
            float pow_5 = rijSquared_5 + softSquared; 
            float pow_6 = rijSquared_6 + softSquared; 
            float pow_7 = rijSquared_7 + softSquared;

            // pow_0 = pow_0 * std::sqrt(pow_0); 
            // pow_1 = pow_1 * std::sqrt(pow_1); 
            // pow_2 = pow_2 * std::sqrt(pow_2); 
            // pow_3 = pow_3 * std::sqrt(pow_3); 
            // pow_4 = pow_4 * std::sqrt(pow_4); 
            // pow_5 = pow_5 * std::sqrt(pow_5); 
            // pow_6 = pow_6 * std::sqrt(pow_6); 
            // pow_7 = pow_7 * std::sqrt(pow_7); 
            
            // printf("\n\n pow\n");
            // printf("{%f, %f, %f, %f, %f, %f, %f, %f}\n",pow_0,pow_1,pow_2,pow_3,pow_4,pow_5,pow_6,pow_7);
            // print_reg(r_pow);
            //const float pow = std::pow(rijSquared + softSquared, 3.f / 2.f);// 2 flops
            
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            mipp::Reg<float> r_ai = grav*r_jm / r_pow; // 3 flops

            float ai_0 = this->G * jm_0 / std::pow(pow_0, 3.f/2.f);
            float ai_1 = this->G * jm_1 / std::pow(pow_1,3.f/2.f);
            float ai_2 = this->G * jm_2 / std::pow(pow_2,3.f/2.f);
            float ai_3 = this->G * jm_3 / std::pow(pow_3,3.f/2.f);
            float ai_4 = this->G * jm_4 / std::pow(pow_4,3.f/2.f);
            float ai_5 = this->G * jm_5 / std::pow(pow_5,3.f/2.f);
            float ai_6 = this->G * jm_6 / std::pow(pow_6,3.f/2.f);
            float ai_7 = this->G * jm_7 / std::pow(pow_7,3.f/2.f);
            
            // printf("\n\n ai\n");
            // printf("{%f, %f, %f, %f, %f, %f, %f, %f}\n",ai_0,ai_1,ai_2,ai_3,ai_4,ai_5,ai_6,ai_7);
            // print_reg(r_ai);

            //const float aj = this->G * d[iBody].m / pow; // 3 flops
           
            /*
              mipp::Reg<float> r_aj = (r_im * r_pow) * this->G; // 3 flops

            */

             
            // add the acceleration value into the acceleration vector: ai += || ai ||.rij

            

            r_ax += (r_ai * rijx);
            r_ay += (r_ai * rijy);
            r_az += (r_ai * rijz);

            
            
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
            this->accelerations.ax[iBody] += mipp::hadd<float>(r_ax);
            this->accelerations.ay[iBody] += mipp::hadd<float>(r_ay);
            this->accelerations.az[iBody] += mipp::hadd<float>(r_az);

        for(unsigned long jBody = loop_tail; jBody < this->getBodies().getN(); jBody++){
            
            const float rijx = d.qx[jBody] - d.qx[iBody]; // 1 flop
            const float rijy = d.qy[jBody] - d.qy[iBody]; // 1 flop
            const float rijz = d.qz[jBody] - d.qz[iBody]; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = rijx * rijx + rijy * rijy + rijz *rijz; // 5 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = this->G * d.m[jBody] / std::pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            this->accelerations.ax[iBody] += ai * rijx; // 2 flops
            this->accelerations.ay[iBody] += ai * rijy; // 2 flops
            this->accelerations.az[iBody] += ai * rijz; // 2 flops
        }
    }
}



void SimulationNBodySIMD::computeBodiesAccelerationV3()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();


    const float softSquared = this->soft *  this->soft;// 1 flops

    //SIMD Internal loop pitch
    constexpr int N = mipp::N<float>();
    unsigned long loop_tail = (this->getBodies().getN()/N)*N;

    //Register with gravity in it
    mipp::Reg<float> grav = this->G;
    
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {

        float ai_x=0,ai_y=0,ai_z=0;
        
        mipp::Reg<float> r_iqx = d.qx[iBody];
        mipp::Reg<float> r_iqy = d.qy[iBody];
        mipp::Reg<float> r_iqz = d.qz[iBody];

        for(unsigned long jj =0; jj < loop_tail; jj+=N){
            mipp::Reg<float> r_jqx = mipp::load(&d.qx[jj]);
            mipp::Reg<float> r_jqy = mipp::load(&d.qy[jj]);
            mipp::Reg<float> r_jqz = mipp::load(&d.qz[jj]);
            mipp::Reg<float> r_jm = mipp::load(&d.m[jj]);

            mipp::Reg<float> rijx = r_jqx - r_iqx; // 1 flop
            mipp::Reg<float> rijy = r_jqy - r_iqy; // 1 flop
            mipp::Reg<float> rijz = r_jqz - r_iqz; // 1 flop
            mipp::Reg<float> rijSquared = rijx * rijx + rijy * rijy + rijz *rijz; // 5 flops
            mipp::Reg<float> r_pow = rijSquared + softSquared;
            r_pow = r_pow * mipp::sqrt<float>(rijSquared+softSquared);
            mipp::Reg<float> ai = (r_jm * this->G)/r_pow;
            ai_x += (mipp::hadd<float>(rijx * ai));
            ai_y += (mipp::hadd<float>(rijy * ai));
            ai_z += (mipp::hadd<float>(rijz * ai));
        }

        for(unsigned long jBody = loop_tail; jBody < this->getBodies().getN(); jBody++){
            
            const float rijx = d.qx[jBody] - d.qx[iBody]; // 1 flop
            const float rijy = d.qy[jBody] - d.qy[iBody]; // 1 flop
            const float rijz = d.qz[jBody] - d.qz[iBody]; // 1 flop

            const float rijSquared = rijx * rijx + rijy * rijy + rijz *rijz; // 5 flops
            const float ai = this->G * d.m[jBody] / std::pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops
            
            ai_x += ai*rijx;
            ai_y += ai*rijy;
            ai_z += ai*rijz;
        }
            this->accelerations.ax[iBody] = ai_x; // 2 flops
            this->accelerations.ay[iBody] = ai_y; // 2 flops
            this->accelerations.az[iBody] = ai_z; // 2 flops
    }
}

void SimulationNBodySIMD::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAccelerationV3();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
