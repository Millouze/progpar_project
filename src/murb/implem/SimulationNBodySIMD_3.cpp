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

#include "SimulationNBodySIMD_3.hpp"

SimulationNBodySIMD_3::SimulationNBodySIMD_3(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 27.f * (((float)this->getBodies().getN()+1) * (float)this->getBodies().getN())/2;
    //this->accelerations.resize(this->getBodies().getN());
    this->accelerations.ax.resize(this->getBodies().getN() + this->getBodies().getPadding());
    this->accelerations.ay.resize(this->getBodies().getN() + this->getBodies().getPadding());
    this->accelerations.az.resize(this->getBodies().getN() + this->getBodies().getPadding());
}

void SimulationNBodySIMD_3::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations.ax.push_back(0.f);
        this->accelerations.ay.push_back(0.f);
        this->accelerations.az.push_back(0.f);
    }
}




void SimulationNBodySIMD_3::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();


    const float softSquared = this->soft *  this->soft;// 1 flops

    //SIMD Internal loop pitch
    constexpr int N = mipp::N<float>();
    

    //Register with gravity in it
    mipp::Reg<float> grav = this->G;
    
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {


        float ai_x = 0.;
        float ai_y = 0.;
        float ai_z = 0.;
        //SIMD registers
        mipp::Reg<float> r_iqx = d.qx[iBody];
        mipp::Reg<float> r_iqy = d.qy[iBody];
        mipp::Reg<float> r_iqz = d.qz[iBody];
        mipp::Reg<float> r_im = d.m[iBody];

        //Normal variables
        float iqx = d.qx[iBody];
        float iqy = d.qy[iBody];
        float iqz = d.qz[iBody];
        float im = d.m[iBody];

        unsigned long loop_tail = ((this->getBodies().getN()-iBody+1)/N)*N+iBody+1;
        mipp::Reg<float> r_ai_x = mipp::set0<float>();
        mipp::Reg<float> r_ai_y = mipp::set0<float>();
        mipp::Reg<float> r_ai_z = mipp::set0<float>();
        
        for(unsigned long jj =iBody+1; jj < loop_tail; jj+=N){
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
            
            r_ai_x += rijx * ai;
            r_ai_y += rijy * ai;
            r_ai_z += rijz * ai;
            //Adding acceleration to jBodies
            // we need mass of iBody and SIMD reg of curent acceleration
            mipp::Reg<float> r_jax = mipp::load(&this->accelerations.ax[jj]);
            mipp::Reg<float> r_jay = mipp::load(&this->accelerations.ay[jj]);
            mipp::Reg<float> r_jaz = mipp::load(&this->accelerations.az[jj]);
            mipp::Reg<float> aj = (r_im * this->G)/r_pow;

            r_jax = r_jax - aj * rijx;
            r_jay = r_jay - aj * rijy;
            r_jaz = r_jaz - aj * rijz;

            mipp::store(&this->accelerations.ax[jj], r_jax);
            mipp::store(&this->accelerations.ay[jj], r_jay);
            mipp::store(&this->accelerations.az[jj], r_jaz);
            // for(unsigned long i = 0; i<N;i++){
            //     float aj = (im * this->G)/r_pow[i];
            //     this->accelerations.ax[jj+i]-= aj*rijx[i];
            //     this->accelerations.ay[jj+i]-= aj*rijy[i];
            //     this->accelerations.az[jj+i]-= aj*rijz[i];
            // }
            
        }
            ai_x += (mipp::hadd<float>(r_ai_x));
            ai_y += (mipp::hadd<float>(r_ai_y));
            ai_z += (mipp::hadd<float>(r_ai_z));

        for(unsigned long jBody = loop_tail; jBody < this->getBodies().getN(); jBody++){
            
            const float rijx = d.qx[jBody] - iqx; // 1 flop
            const float rijy = d.qy[jBody] - iqy; // 1 flop
            const float rijz = d.qz[jBody] - iqz; // 1 flop

            const float rijSquared = rijx * rijx + rijy * rijy + rijz *rijz; // 5 flops
            const float pow = std::pow(rijSquared + softSquared, 3.f / 2.f);// 2 flops
            const float ai = this->G * d.m[jBody] / pow; // 5 flops
            
            ai_x += ai*rijx;
            ai_y += ai*rijy;
            ai_z += ai*rijz;

            //Adding acceleration to jBodies
            
            const float aj = this->G * im / pow; // 3 flops
            this->accelerations.ax[jBody] -= aj * rijx; // 2 flops
            this->accelerations.ay[jBody] -= aj * rijy; // 2 flops
            this->accelerations.az[jBody] -= aj * rijz; // 2 flops
        }
            this->accelerations.ax[iBody] += ai_x; // 2 flops
            this->accelerations.ay[iBody] += ai_y; // 2 flops
            this->accelerations.az[iBody] += ai_z; // 2 flops
    }
}

void SimulationNBodySIMD_3::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
