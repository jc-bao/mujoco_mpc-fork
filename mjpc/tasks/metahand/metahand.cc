// Task definition file which use Metahand for in-hand orientation
// Author: Chaoyi Pan
// Date: June 18, 2025

#include "mjpc/tasks/metahand/metahand.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {
std::string Metahand::XmlPath() const {
  return GetModelPath("metahand/task.xml");
}
std::string Metahand::Name() const { return "Metahand"; }

// ------- Residuals for cube manipulation task ------
//     Cube position: (3)
//     Cube orientation: (3)
//     Cube linear velocity: (3)
//     Control: (16), there are 16 servos
//     Nominal pose: (16)
//     Joint velocity: (16)
// ------------------------------------------
void Metahand::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                   double *residual) const {
  int counter = 0;

  // ---------- Cube position ----------
  // double *object_position = SensorByName(model, data, "object_position");

  // ---------- Cube orientation ----------
  // double *object_orientation = SensorByName(model, data, "object_orientation");

  // ---------- Cube linear velocity ----------
  // double *object_linear_velocity =
  //     SensorByName(model, data, "object_linear_velocity");

  // mju_copy(residual + counter, object_linear_velocity, 3);
  // counter += 3;

  // ---------- Control ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  // ---------- Nominal Pose ----------
  mju_sub(residual + counter, data->qpos + 7, model->key_qpos + 7, 16);
  counter += 16;

  // ---------- Joint Velocity ----------
  mju_copy(residual + counter, data->qvel + 6, 16);
  counter += 16;

  // Sanity check
  CheckSensorDim(model, counter);
}

}  // namespace mjpc
