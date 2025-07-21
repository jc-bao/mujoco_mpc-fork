#ifndef MJPC_TASKS_G1_G1_H_
#define MJPC_TASKS_G1_G1_H_

#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class G1 : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const G1* task) : BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  G1() : residual_(this) { LoadTrajectoryData(); }

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  
  // Trajectory data
  std::vector<std::vector<float>> qpos_trajectory_;  // (914, 30)
  std::vector<std::vector<float>> qvel_trajectory_;  // (914, 29)
  double trajectory_dt_;  // time step between trajectory points
  
  // Load trajectory data from binary files
  void LoadTrajectoryData();
  std::vector<std::vector<float>> LoadBinaryArray(const std::string& filename, int rows, int cols);
};
}  // namespace mjpc

#endif  // MJPC_TASKS_G1_G1_H_