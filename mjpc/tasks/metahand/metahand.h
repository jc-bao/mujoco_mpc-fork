// Task definition file which use Metahand for in-hand orientation
// Author: Chaoyi Pan
// Date: June 18, 2025

#ifndef MJPC_TASKS_METAHAND_METAHAND_H_
#define MJPC_TASKS_METAHAND_METAHAND_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class Metahand : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const Metahand *task) : BaseResidualFn(task) {}
    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;
  };
  Metahand() : residual_(this) {}

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_METAHAND_METAHAND_H_
