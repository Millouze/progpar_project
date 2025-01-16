#ifndef SIMULATION_N_BODY_AOS_HPP_
#define SIMULATION_N_BODY_AOS_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyCUDA_AoS : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyCUDA_AoS(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyCUDA_AoS() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
