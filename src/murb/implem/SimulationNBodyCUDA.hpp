#ifndef SIMULATION_N_BODY_CUDA_HPP_
#define SIMULATION_N_BODY_CUDA_HPP_

#include <cuda_runtime_api.h>
#include <string>

#include "core/SimulationNBodyInterface.hpp"
#include <cuda.h>

class SimulationNBodyCUDA : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    dim3 blocksPerGrid = {(unsigned int)this->getBodies().getN() / 1024};
    dim3 threadsPerBlock = {1024};
    float *d_qx, *d_qy, *d_qz, *d_m;
    accAoS_t<float> *d_accelerations;
    
  public:
    SimulationNBodyCUDA(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    ~SimulationNBodyCUDA() {
      cudaFree(d_qx);
      cudaFree(d_qy);
      cudaFree(d_qz);
      cudaFree(d_m);
      cudaFree(d_accelerations);
    }
    virtual void computeOneIteration();

  protected:
    void initIteration();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
