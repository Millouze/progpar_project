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

#include "SimulationNBodyOpenCL.hpp"

SimulationNBodyOpenCL::SimulationNBodyOpenCL(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 27.f * (((float)this->getBodies().getN()+1) * (float)this->getBodies().getN())/2;
    //this->accelerations.resize(this->getBodies().getN());
    this->accelerations.ax.resize(this->getBodies().getN() + this->getBodies().getPadding());
    this->accelerations.ay.resize(this->getBodies().getN() + this->getBodies().getPadding());
    this->accelerations.az.resize(this->getBodies().getN() + this->getBodies().getPadding());
}

void SimulationNBodyOpenCL::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations.ax.push_back(0.f);
        this->accelerations.ay.push_back(0.f);
        this->accelerations.az.push_back(0.f);
    }
}




void SimulationNBodyOpenCL::computeBodiesAcceleration()
{
    //what now ?
}

void SimulationNBodyOpenCL::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
