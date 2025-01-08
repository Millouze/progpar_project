
#ifndef SIMULATION_N_BODY_OpenCL_HPP_
#define SIMULATION_N_BODY_OpenCL_HPP_

#include <string>

#include "core/Bodies.hpp"
#include "core/SimulationNBodyInterface.hpp"

class  SimulationNBodyOpenCL : public SimulationNBodyInterface {
  protected:
    //std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    accSoA_t<float> accelerations; /*!< Structure of arrays of body acceleration. */
  public:
    SimulationNBodyOpenCL(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyOpenCL() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
