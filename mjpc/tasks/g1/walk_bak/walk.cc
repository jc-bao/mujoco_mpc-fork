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

  // ----- torso height ----- //
  double torso_height = SensorByName(model, data, "torso_position")[2];
  residual[counter++] = torso_height - parameters_[0];

  // ----- pelvis / feet ----- //
  double* foot_right = SensorByName(model, data, "foot_right");
  double* foot_left = SensorByName(model, data, "foot_left");
  double pelvis_height = SensorByName(model, data, "pelvis_position")[2];
  residual[counter++] =
      0.5 * (foot_left[2] + foot_right[2]) - pelvis_height - 0.2;

  // ----- balance ----- //
  // capture point
  double* subcom = SensorByName(model, data, "torso_subcom");
  double* subcomvel = SensorByName(model, data, "torso_subcomvel");

  double capture_point[3];
  mju_addScl(capture_point, subcom, subcomvel, 0.3, 3);
  capture_point[2] = 1.0e-3;

  // project onto line segment

  double axis[3];
  double center[3];
  double vec[3];
  double pcp[3];
  mju_sub3(axis, foot_right, foot_left);
  axis[2] = 1.0e-3;
  double length = 0.5 * mju_normalize3(axis) - 0.05;
  mju_add3(center, foot_right, foot_left);
  mju_scl3(center, center, 0.5);
  mju_sub3(vec, capture_point, center);

  // project onto axis
  double t = mju_dot3(vec, axis);

  // clamp
  t = mju_max(-length, mju_min(length, t));
  mju_scl3(vec, axis, t);
  mju_add3(pcp, vec, center);
  pcp[2] = 1.0e-3;

  // is standing
  double standing =
      torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;

  mju_sub(&residual[counter], capture_point, pcp, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);

  counter += 2;

  // ----- upright ----- //
  double* torso_up = SensorByName(model, data, "torso_up");
  double* pelvis_up = SensorByName(model, data, "pelvis_up");
  double* foot_right_up = SensorByName(model, data, "foot_right_up");
  double* foot_left_up = SensorByName(model, data, "foot_left_up");
  double z_ref[3] = {0.0, 0.0, 1.0};

  // torso
  residual[counter++] = torso_up[2] - 1.0;

  // pelvis
  residual[counter++] = 0.3 * (pelvis_up[2] - 1.0);

  // right foot
  mju_sub3(&residual[counter], foot_right_up, z_ref);
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;

  mju_sub3(&residual[counter], foot_left_up, z_ref);
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;

  // ----- posture ----- //
  mju_copy(&residual[counter], data->qpos + 7, model->nq - 7);
  counter += model->nq - 7;

  // ----- position ----- //
  double* torso_pos = SensorByName(model, data, "torso_position");
  // get target position to {0, 0, 0}
  double target[3] = {0, 0, 0};
  residual[counter++] = torso_pos[0] - target[0];
  residual[counter++] = torso_pos[1] - target[1];

  // com vel
  // double* waist_lower_subcomvel =
  //     SensorByName(model, data, "waist_lower_subcomvel");
  // double* torso_velocity = SensorByName(model, data, "torso_velocity");
  // double com_vel[2];
  // mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
  // mju_scl(com_vel, com_vel, 0.5, 2);

  // ----- move feet ----- //
  // double* foot_right_vel = SensorByName(model, data, "foot_right_velocity");
  // double* foot_left_vel = SensorByName(model, data, "foot_left_velocity");
  // double move_feet[2];
  // mju_copy(move_feet, com_vel, 2);
  // mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
  // mju_addToScl(move_feet, foot_left_vel, -0.5, 2);

  // mju_copy(&residual[counter], move_feet, 2);
  // mju_scl(&residual[counter], &residual[counter], standing, 2);
  // counter += 2;

  // ----- control ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ----- gait ----- //
  for (int i = 0; i < 2; i++) {
    // TODO: make this a parameter
    double amplitude = 0.05; 
    double duty_ratio = 0.5;
    double footphase = 0.0;
    double currentphase = data->time * 2.0; // 2.0 is the gait frequency
    if (i == 0) {
      footphase = 0.0;
    } else if (i == 1) {
      footphase = mjPI; 
    }
    double angle = fmod(currentphase + mjPI - footphase, 2 * mjPI) - mjPI;
    double value = 0;
    if (duty_ratio < 1) {
      angle *= 0.5 / (1 - duty_ratio);
      value = amplitude * mju_cos(mju_clip(angle, -mjPI / 2, mjPI / 2));
    }
    double foot_height = 0;
    if (i == 0) {
      double* foot_right_pos = SensorByName(model, data, "right_foot");
      foot_height = foot_right_pos[2];
    } else {
      double* foot_left_pos = SensorByName(model, data, "left_foot");
      foot_height = foot_left_pos[2];
    }
    double height_difference = mju_max(0.0, value - foot_height);
    residual[counter++] = height_difference;
  }

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
