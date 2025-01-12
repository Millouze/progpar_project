#ifndef SIMULATION_N_BODY_AOS_HPP_
#define SIMULATION_N_BODY_AOS_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyAoS : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyAoS(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyAoS() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
