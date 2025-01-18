#ifndef SIMULATION_N_BODY_HYBRID_HPP_
#define SIMULATION_N_BODY_HYBRID_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class  SimulationNBodyHybrid: public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    float *d_qx, *d_qy, *d_qz, *d_m;
    accAoS_t<float> *d_accelerations;
  public:
    SimulationNBodyHybrid(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);

    ~SimulationNBodyHybrid(); 
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAccelerationCPU();

  private:
    unsigned int NforWrap;
    int NforProcs;
};

#endif /* SIMULATION_N_BODY__HPP_ */
