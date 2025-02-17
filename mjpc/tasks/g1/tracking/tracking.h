#ifndef MJPC_TASKS_G1_TRACKING_TASK_H_
#define MJPC_TASKS_G1_TRACKING_TASK_H_

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace g1 {

class Tracking : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Tracking* task, int current_mode = 0,
                        double reference_time = 0)
        : mjpc::BaseResidualFn(task),
          current_mode_(current_mode),
          reference_time_(reference_time) {}

    // ------------- Residuals for G1 tracking task -------------
    //   Number of residuals:
    //     Residual (0): Joint vel: minimise joint velocity
    //     Residual (1): Control: minimise control
    //     Residual (2-11): Tracking position: minimise tracking position error
    //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
    //     Residual (11-20): Tracking velocity: minimise tracking velocity error
    //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
    //   Number of parameters: 0
    // ----------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
   private:
    friend class Tracking;
    int current_mode_;
    double reference_time_;
  };

  Tracking() : residual_(this) {}

  // --------------------- Transition for G1 task ------------------------
  //   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
  //   Linearly interpolate between two consecutive key frames in order to
  //   smooth the transitions between keyframes.
  // ---------------------------------------------------------------------------
  void TransitionLocked(mjModel* model, mjData* data) override;

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.current_mode_,
                                        residual_.reference_time_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace g1
}  // namespace mjpc

#endif  // MJPC_TASKS_G1_TRACKING_TASK_H_