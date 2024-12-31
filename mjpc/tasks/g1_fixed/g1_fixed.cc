#include "mjpc/tasks/g1_fixed/g1_fixed.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::g1_fixed
{

  std::string G1Fixed::XmlPath() const
  {
    return GetModelPath("g1_fixed/task.xml");
  }
  std::string G1Fixed::Name() const { return "G1 Fixed"; }

  // ------------------ Residuals for G1 Fixed task ------------
  //   Number of residuals: 6
  //     Residual (0): Control: minimise control
  //     Residual (1): Joint vel: minimise joint velocity
  //     Residual (2): Joint pos: minimise joint position
  //   Number of parameters: 1
  //     Parameter (0): height_goal
  // ----------------------------------------------------------------
  void G1Fixed::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                   double *residual) const
  {
    int counter = 0;

    // ----- joint velocity ----- //
    mju_copy(residual + counter, data->qvel, model->nv);
    counter += model->nv;

    // ----- action ----- //
    mju_copy(&residual[counter], data->ctrl, model->nu);
    counter += model->nu;

    // ----- Posture ----- //
    double *home = KeyQPosByName(model, data, "stand");
    mju_sub(residual + counter, data->qpos, home, model->nq);
    counter += model->nq;

    // sensor dim sanity check
    int user_sensor_dim = 0;
    for (int i = 0; i < model->nsensor; i++)
    {
      if (model->sensor_type[i] == mjSENS_USER)
      {
        user_sensor_dim += model->sensor_dim[i];
      }
    }
    if (user_sensor_dim != counter)
    {
      mju_error_i(
          "mismatch between total user-sensor dimension "
          "and actual length of residual %d",
          counter);
    }
  }

} // namespace mjpc::g1_fixed
