#ifndef SIMULATION_N_BODY_OPENCL_HPP_
#define SIMULATION_N_BODY_OPENCL_HPP_

#include <string>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyOpenCL : public SimulationNBodyInterface {
  protected:
    accSoA_t<float> accelerations;
    cl_context context;
    cl_kernel kernel;
    cl_command_queue cmd_queue;
    cl_mem in_buf_qx;
    cl_mem in_buf_qy;
    cl_mem in_buf_qz;
    cl_mem in_buf_m;
    cl_mem out_buf_ax;
    cl_mem out_buf_ay;
    cl_mem out_buf_az;
    unsigned long boundary;
    
  public:
    SimulationNBodyOpenCL(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyOpenCL() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_OPENCL_HPP_ */
