#ifndef SIMULATION_N_BODY_SIMD_HPP_
#define SIMULATION_N_BODY_SIMD_HPP_

#include <string>

#include "core/Bodies.hpp"
#include "core/SimulationNBodyInterface.hpp"

class  SimulationNBodySIMD : public SimulationNBodyInterface {
  protected:
    //std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    accSoA_t<float> accelerations; /*!< Structure of arrays of body acceleration. */
  public:
    SimulationNBodySIMD(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodySIMD() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
    void computeBodiesAccelerationV2();
    void computeBodiesAccelerationV3();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
