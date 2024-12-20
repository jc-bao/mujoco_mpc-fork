#ifndef MJPC_TASKS_G1_WALK_TASK_H_
#define MJPC_TASKS_G1_WALK_TASK_H_

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace g1 {

class Walk : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Walk* task) : mjpc::BaseResidualFn(task) {}

    // ------------------ Residuals for humanoid walk task ------------
    //   Number of residuals:
    //     Residual (0): torso height
    //     Residual (1): pelvis-feet aligment
    //     Residual (2): balance
    //     Residual (3): upright
    //     Residual (4): posture
    //     Residual (5): walk
    //     Residual (6): move feet
    //     Residual (7): control
    //   Number of parameters:
    //     Parameter (0): torso height goal
    //     Parameter (1): speed goal
    // ----------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  Walk() : residual_(this) {}

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace g1
}  // namespace mjpc

#endif  // MJPC_TASKS_G1_WALK_TASK_H_