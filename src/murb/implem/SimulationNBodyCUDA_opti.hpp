#ifndef SIMULATION_N_BODY_CUDA_OPTI_HPP_
#define SIMULATION_N_BODY_CUDA_OPTI_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyCUDA_opti : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    
    float *d_qx, *d_qy, *d_qz, *d_m;
    accAoS_t<float> *d_accelerations;
    
  public:
    SimulationNBodyCUDA_opti(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    ~SimulationNBodyCUDA_opti(); 
    virtual void computeOneIteration();

  protected:
    void initIteration();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
