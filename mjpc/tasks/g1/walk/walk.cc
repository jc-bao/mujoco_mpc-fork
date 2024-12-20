#include "mjpc/tasks/g1/walk/walk.h"

#include <iostream>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc::g1 {
std::string Walk::XmlPath() const {
  return GetModelPath("g1/walk/task.xml");
}
std::string Walk::Name() const { return "G1 Walk"; }

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
void Walk::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                double* residual) const {
  int counter = 0;

  // ----- upright ----- //
  double* torso_up = SensorByName(model, data, "torso_up");
  double* pelvis_up = SensorByName(model, data, "pelvis_up");
  double* foot_right_up = SensorByName(model, data, "foot_right_up");
  double* foot_left_up = SensorByName(model, data, "foot_left_up");
  // torso
  residual[counter++] = torso_up[2] - 1.0;
  // pelvis
  residual[counter++] = 0.3 * (pelvis_up[2] - 1.0);
  // right foot
  residual[counter++] = 1.0 * (foot_right_up[2] - 1.0);
  // left foot
  residual[counter++] = 1.0 * (foot_left_up[2] - 1.0);

  // ----- torso height ----- //
  double height_goal = parameters_[0];
  double torso_height = SensorByName(model, data, "torso_position")[2];
  residual[counter++] = torso_height - height_goal;

  // ----- position ----- //
  double* torso_pos = SensorByName(model, data, "torso_position");
  // get target position to {0, 0, 0}
  double target[3] = {0, 0, 0};
  residual[counter++] = torso_pos[0] - target[0];
  residual[counter++] = torso_pos[1] - target[1];
  residual[counter++] = 0.0;

  // ----- gait ----- //
  double* foot_right_pos = SensorByName(model, data, "right_foot_position");
  double* foot_left_pos = SensorByName(model, data, "left_foot_position");
  double avg_foot_pos[3];
  mju_add3(avg_foot_pos, foot_right_pos, foot_left_pos);
  mju_scl3(avg_foot_pos, avg_foot_pos, 0.5);
  for (int i = 0; i < 2; i++) {
    // TODO: make this a parameter
    double amplitude = 0.05; 
    double duty_ratio = 0.5;
    double footphase = 0.0;
    if (i == 0) {
      footphase = 0.0;
    } else if (i == 1) {
      footphase = mjPI; 
    }
    double currentphase = data->time * 1.0 * mjPI; // 2.0 is the gait frequency
    double angle = fmod(currentphase + mjPI - footphase, 2 * mjPI) - mjPI;
    double target_foot_height = 0;
    if (duty_ratio < 1) {
      angle *= 0.5 / (1 - duty_ratio);
      target_foot_height = amplitude * mju_cos(mju_clip(angle, -mjPI / 2, mjPI / 2));
    }
    double foot_height = 0;
    if (i == 0) {
      foot_height = foot_right_pos[2];
    } else {
      foot_height = foot_left_pos[2];
    }
    residual[counter++] = target_foot_height - foot_height;
  }

  // ----- balance ----- //
  double* compos = SensorByName(model, data, "pelvis_subcom");
  double* comvel = SensorByName(model, data, "pelvis_subcomvel");
  double capture_point[3];
  double fall_time = mju_sqrt(2*height_goal / 9.81);
  mju_addScl3(capture_point, compos, comvel, fall_time);
  residual[counter++] = capture_point[0] - avg_foot_pos[0];
  residual[counter++] = capture_point[1] - avg_foot_pos[1];

  // ----- effort ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ----- posture ----- //
  double* home = KeyQPosByName(model, data, "stand");
  mju_sub(residual + counter, data->qpos + 7, home + 7, model->nu);
  counter += model->nu;

  // ----- yaw ----- //
  int torso_body_id = mj_name2id(model, mjOBJ_BODY, "torso");
  double* torso_xmat = data->xmat + 9 * torso_body_id;
  double torso_heading[2] = {torso_xmat[0], torso_xmat[3]};
  mju_normalize(torso_heading, 2);
  // TODO: make this a parameter
  double heading_goal = 0.0;
  residual[counter++] = torso_heading[0] - mju_cos(heading_goal);
  residual[counter++] = torso_heading[1] - mju_sin(heading_goal);

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

}  // namespace mjpc::g1
