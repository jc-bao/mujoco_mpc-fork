#include "mjpc/tasks/h1_mani/pnp/pnp.h"

#include <iostream>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc::h1_mani
{
  std::string PNP::XmlPath() const
  {
    return GetModelPath("h1_mani/pnp/task.xml");
  }
  std::string PNP::Name() const { return "H1 PNP"; }

  // ------------------ Residuals for humanoid pnp task ------------
  //   Number of residuals:
  //     Residual (0): torso height
  //     Residual (1): pelvis-feet aligment
  //     Residual (2): balance
  //     Residual (3): upright
  //     Residual (4): posture
  //     Residual (5): pnp
  //     Residual (6): move feet
  //     Residual (7): control
  //   Number of parameters:
  //     Parameter (0): torso height goal
  //     Parameter (1): speed goal
  // ----------------------------------------------------------------
  void PNP::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                 double *residual) const
  {
    int counter = 0;


    double gait_selection = parameters_[5];
    // if 0, stand, if 1, walk
    int gait_mode = ReinterpretAsInt(gait_selection);

    // ----- upright ----- //
    double *torso_up = SensorByName(model, data, "torso_up");
    double *pelvis_up = SensorByName(model, data, "pelvis_up");
    double *foot_right_up = SensorByName(model, data, "foot_right_up");
    double *foot_left_up = SensorByName(model, data, "foot_left_up");
    // torso
    residual[counter++] = torso_up[2] - 1.0;
    // pelvis
    residual[counter++] = 0.3 * (pelvis_up[2] - 1.0);
    // foot
    residual[counter++] = 1.0 * (foot_right_up[2] - 1.0);
    residual[counter++] = 1.0 * (foot_left_up[2] - 1.0);
    // ankle joint angle
    double *hip_y_right = SensorByName(model, data, "hip_y_right");
    double *hip_y_left = SensorByName(model, data, "hip_y_left");
    double *knee_right = SensorByName(model, data, "knee_right");
    double *knee_left = SensorByName(model, data, "knee_left");
    double *ankle_y_right = SensorByName(model, data, "ankle_y_right");
    double *ankle_y_left = SensorByName(model, data, "ankle_y_left");
    residual[counter++] = hip_y_right[0] + knee_right[0] + ankle_y_right[0];
    residual[counter++] = hip_y_left[0] + knee_left[0] + ankle_y_left[0];

    // ----- torso height ----- //
    double height_goal = parameters_[0];
    double torso_height = SensorByName(model, data, "torso_position")[2];
    residual[counter++] = torso_height - height_goal;

    // torso velocity
    double* torso_vel = SensorByName(model, data, "torso_velocity");
    residual[counter++] = torso_vel[0];
    residual[counter++] = torso_vel[1];
    residual[counter++] = torso_vel[2];

    // ----- position ----- //
    int goal_id = mj_name2id(model, mjOBJ_XBODY, "goal");
    int goal_mocap_id = model->body_mocapid[goal_id];
    if (goal_mocap_id < 0)
      mju_error("body 'goal' is not mocap");
    double *goal = data->mocap_pos + 3 * goal_mocap_id;
    double *torso_pos = SensorByName(model, data, "torso_position");
    double torso_to_goal[2];
    mju_sub(torso_to_goal, goal, torso_pos, 2);
    // double dist2goal = mju_norm(torso_to_goal, 2);
    if (gait_mode == 0)
    {
      residual[counter++] = 0.0;
      residual[counter++] = 0.0;
      residual[counter++] = 0.0;
    }
    else
    {
      // get target position to {0, 0, 0}
      residual[counter++] = torso_to_goal[0];
      residual[counter++] = torso_to_goal[1];
      residual[counter++] = 0.0;
    }

    // ----- yaw ----- //
    int torso_body_id = mj_name2id(model, mjOBJ_BODY, "torso");
    double *torso_xmat = data->xmat + 9 * torso_body_id;
    double torso_heading[2] = {torso_xmat[0], torso_xmat[3]};
    mju_normalize(torso_heading, 2);
    double heading_goal = parameters_[8];
    double target_heading[2] = {mju_cos(heading_goal), mju_sin(heading_goal)};
    if (gait_mode == 0) {
      residual[counter++] = 0.0;
      residual[counter++] = 0.0;
    } else {
      residual[counter++] = torso_heading[0] - target_heading[0];
      residual[counter++] = torso_heading[1] - target_heading[1];
    }
    int pelvis_body_id = mj_name2id(model, mjOBJ_BODY, "pelvis");
    double *pelvis_xmat = data->xmat + 9 * pelvis_body_id;
    double pelvis_heading[2] = {pelvis_xmat[0], pelvis_xmat[3]};
    mju_normalize(pelvis_heading, 2);
    if (gait_mode == 0) {
      residual[counter++] = 0.0;
      residual[counter++] = 0.0;
    } else {
      residual[counter++] = pelvis_heading[0] - target_heading[0];
      residual[counter++] = pelvis_heading[1] - target_heading[1];
    }

    // ----- gait ----- //
    // double *pelvis_pos = SensorByName(model, data, "pelvis_position");
    double pelvis_heading_ortho[2] = {-pelvis_heading[1], pelvis_heading[0]};
    mju_normalize(pelvis_heading, 2);
    mju_normalize(pelvis_heading_ortho, 2);
    double *toe_right_pos = SensorByName(model, data, "tracking_pos[rtoe]");
    double *heel_right_pos = SensorByName(model, data, "tracking_pos[rheel]");
    double *toe_left_pos = SensorByName(model, data, "tracking_pos[ltoe]");
    double *heel_left_pos = SensorByName(model, data, "tracking_pos[lheel]");
    // TODO: make this a parameter
    double amplitude = parameters_[11];
    // set amplitude to 0 if torso_pos - goal is smaller than 0.1 
    if (gait_mode == 0)
    {
      amplitude = 0.0;
    }
    double duty_ratio = parameters_[9];
    double footphase = 0.0;
    double frequency = parameters_[10];
    double foot_y_distance_target = parameters_[6];
    double foot_x_distance_target = parameters_[7];
    double left_toe_cost[3] = {0.0, 0.0, 0.0};
    double left_heel_cost[3] = {0.0, 0.0, 0.0};
    double right_toe_cost[3] = {0.0, 0.0, 0.0};
    double right_heel_cost[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 2; i++) // 0: left, 1: right
    {
      if (i == 0)
      {
        footphase = 0.0;
      }
      else if (i == 1)
      {
        footphase = mjPI;
      }
      double currentphase = data->time * mjPI * frequency; // 2.0 is the gait frequency
      double angle = fmod(currentphase + mjPI - footphase, 2 * mjPI) - mjPI;
      double target_foot_height = 0.01;
      if (duty_ratio < 1)
      {
        angle *= 0.5 / (1 - duty_ratio);
        target_foot_height += amplitude * mju_cos(mju_clip(angle, -mjPI / 2, mjPI / 2));
        // target_foot_height += amplitude * 0.5 * (mju_cos(2.0 * mju_clip(angle, -mjPI / 2, mjPI / 2)) + 1.0);
      }
      // z position of the foot
      double *frame_pos = nullptr;
      double x_tar = 0.0;
      double y_tar = 0.0;
      for (int j = 0; j < 2; j++) // 0: toe, 1: heel
      {
        if (i == 0) // left foot
        {
          // right foot
          if (j == 0) // toe
          {
            frame_pos = toe_right_pos;
            x_tar = foot_x_distance_target + 0.17 - 0.045;
          }
          else
          {
            frame_pos = heel_right_pos;
            x_tar = foot_x_distance_target - 0.08 - 0.045;
          }
          y_tar = -foot_y_distance_target;
        }
        else // right foot
        {
          if (j == 0) // toe
          {
            frame_pos = toe_left_pos;
            x_tar = foot_x_distance_target + 0.17 - 0.045;
          }
          else
          {
            frame_pos = heel_left_pos;
            x_tar = foot_x_distance_target - 0.08 - 0.045;
          }
          y_tar = foot_y_distance_target;
        }
        double frame_pos_world[3];
        double *compos_pelvis = SensorByName(model, data, "pelvis_subcom");
        mju_sub3(frame_pos_world, frame_pos, compos_pelvis);
        double frame_pos_pelvis[2];
        frame_pos_pelvis[0] = mju_dot(pelvis_heading, frame_pos_world, 2);
        frame_pos_pelvis[1] = mju_dot(pelvis_heading_ortho, frame_pos_world, 2);
        if (i == 0)
        {
          if (j == 0)
          {
            right_toe_cost[0] = 1.0 * (frame_pos_pelvis[0] - x_tar);
            right_toe_cost[1] = 1.0 * (frame_pos_pelvis[1] - y_tar);
            right_toe_cost[2] = 1.0 * (frame_pos[2] - target_foot_height);
          }
          else
          {
            right_heel_cost[0] = 1.0 * (frame_pos_pelvis[0] - x_tar);
            right_heel_cost[1] = 1.0 * (frame_pos_pelvis[1] - y_tar);
            right_heel_cost[2] = 1.0 * (frame_pos[2] - target_foot_height);
          }
        }
        else
        {
          if (j == 0)
          {
            left_toe_cost[0] = 1.0 * (frame_pos_pelvis[0] - x_tar);
            left_toe_cost[1] = 1.0 * (frame_pos_pelvis[1] - y_tar);
            left_toe_cost[2] = 1.0 * (frame_pos[2] - target_foot_height);
          }
          else
          {
            left_heel_cost[0] = 1.0 * (frame_pos_pelvis[0] - x_tar);
            left_heel_cost[1] = 1.0 * (frame_pos_pelvis[1] - y_tar);
            left_heel_cost[2] = 1.0 * (frame_pos[2] - target_foot_height);
          }
        }
      }
    }
    double gait_scale = 1.0;
    for (int i = 0; i < 3; i++)
    {
      if (i == 2) {
        if (gait_mode == 0) {
          gait_scale = 0.1;
        } else {
          gait_scale = 1.0;
        }
      }
      residual[counter++] = left_heel_cost[i] * gait_scale;
      residual[counter++] = left_toe_cost[i] * gait_scale;
      residual[counter++] = right_heel_cost[i] * gait_scale;
      residual[counter++] = right_toe_cost[i] * gait_scale;
    }

    // ----- balance ----- //
    double *foot_right_pos = SensorByName(model, data, "right_foot_position");
    double *foot_left_pos = SensorByName(model, data, "left_foot_position");
    double avg_foot_pos[3];
    mju_add3(avg_foot_pos, foot_right_pos, foot_left_pos);
    mju_scl3(avg_foot_pos, avg_foot_pos, 0.5);
    double *compos = SensorByName(model, data, "pelvis_subcom");
    double *comvel = SensorByName(model, data, "pelvis_subcomvel");
    double capture_point[3];
    double fall_time = mju_sqrt(2 * height_goal / 9.81);
    mju_addScl3(capture_point, compos, comvel, fall_time);
    residual[counter++] = capture_point[0] - avg_foot_pos[0];
    residual[counter++] = capture_point[1] - avg_foot_pos[1];

    // ----- effort ----- //
    mju_scl(residual + counter, data->actuator_force, 2e-2, model->nu);
    // make pitch motor effort scale to 0.3
    // mju_scl(residual + counter + 2, residual + counter + 2, 0.3, 3);
    // mju_scl(residual + counter + 5 + 2, residual + counter + 5 + 2, 0.3, 3);
    counter += model->nu;

    // joint velocity
    mju_scl(residual + counter, data->qvel + 6, 1e-2, model->nv - 6);
    counter += (model->nv - 6);

    // ----- posture ----- //
    double *home = KeyQPosByName(model, data, "stand");
    mju_sub(residual + counter, data->qpos + 7, home + 7, model->nu);
    // double upper_body_posture_scale = parameters_[1]; 
    int n_motor_per_leg = 5;
    int n_hip_motor = 3;
    int n_upper_body_motor = 9;
    double hip_pitch_motor_scale = parameters_[2];
    mju_scl(residual + counter + 2, residual + counter + 2, hip_pitch_motor_scale, 1);
    mju_scl(residual + counter + n_motor_per_leg + 2, residual + counter + n_motor_per_leg + 2, hip_pitch_motor_scale, 1);
    double hip_yaw_roll_motor_scale = parameters_[12];
    mju_scl(residual + counter, residual + counter, hip_yaw_roll_motor_scale, n_hip_motor-1);
    mju_scl(residual + counter + n_motor_per_leg, residual + counter + n_motor_per_leg, hip_yaw_roll_motor_scale, n_hip_motor-1);
    double knee_motor_scale = parameters_[3];
    mju_scl(residual + counter + n_hip_motor, residual + counter + n_hip_motor, knee_motor_scale, 1);
    mju_scl(residual + counter + n_motor_per_leg + n_hip_motor, residual + counter + n_motor_per_leg + n_hip_motor, knee_motor_scale, 1);
    double ankle_motor_scale = parameters_[4];
    mju_scl(residual + counter + n_motor_per_leg - 1, residual + counter + n_motor_per_leg - 1, ankle_motor_scale, 1);
    mju_scl(residual + counter + 2 * n_motor_per_leg - 1, residual + counter + 2 * n_motor_per_leg - 1, ankle_motor_scale, 1);
    double upper_body_motor_scale = parameters_[1];
    if (upper_body_motor_scale > 0) {
      mju_scl(residual + counter + 2 * n_motor_per_leg, residual + counter + 2 * n_motor_per_leg, upper_body_motor_scale, n_upper_body_motor);
    }
    counter += model->nu;

    // ----- linear velocity ----- //
    mju_copy3(residual + counter, SensorByName(model, data, "torso_linvel"));
    counter += 3;

    // ----- angular momentum ----- //
    mju_copy3(residual + counter, SensorByName(model, data, "torso_angmom"));
    counter += 3;

    // ----- hand reach ----- //
    // get mode
    double hand_reach_mode_selection = parameters_[13];
    // mode=0 1 2
    int hand_reach_mode = ReinterpretAsInt(hand_reach_mode_selection);
    double *left_hand_front = SensorByName(model, data, "left_hand_front");
    double *left_hand_back = SensorByName(model, data, "left_hand_back");
    double *right_hand_front = SensorByName(model, data, "right_hand_front");
    double *right_hand_back = SensorByName(model, data, "right_hand_back");
    double *left_box_front = nullptr;
    double *right_box_front = nullptr;
    double *left_box_back = nullptr;
    double *right_box_back = nullptr;
    if (hand_reach_mode == 0) {
      left_box_front = SensorByName(model, data, "box1-1");
      right_box_front = SensorByName(model, data, "box1-2");
      left_box_back = SensorByName(model, data, "box1-3");
      right_box_back = SensorByName(model, data, "box1-4");
    }
    else if (hand_reach_mode == 1) {
      left_box_front = SensorByName(model, data, "box2-1");
      right_box_front = SensorByName(model, data, "box2-2");
      left_box_back = SensorByName(model, data, "box2-3");
      right_box_back = SensorByName(model, data, "box2-4");
    }
    else if (hand_reach_mode == 2) {
      left_box_front = SensorByName(model, data, "box3-1");
      right_box_front = SensorByName(model, data, "box3-2");
      left_box_back = SensorByName(model, data, "box3-3");
      right_box_back = SensorByName(model, data, "box3-4");
    }
    else if (hand_reach_mode == 3) {
      left_box_front = SensorByName(model, data, "box4-1");
      right_box_front = SensorByName(model, data, "box4-2");
      left_box_back = SensorByName(model, data, "box4-3");
      right_box_back = SensorByName(model, data, "box4-4");
    }
    else {
      mju_error("Invalid hand reach mode: %d", hand_reach_mode);
    }
    mju_sub3(residual + counter, left_hand_front, left_box_front);
    counter += 3;
    mju_sub3(residual + counter, left_hand_back, left_box_back);
    counter += 3;
    mju_sub3(residual + counter, right_hand_front, right_box_front);
    counter += 3;
    mju_sub3(residual + counter, right_hand_back, right_box_back);
    counter += 3;

    // ----- target reach ----- //
    // double *left_target = SensorByName(model, data, "left_target");
    // double *right_target = SensorByName(model, data, "right_target");
    // mju_sub3(residual + counter, left_box_front, left_target);
    // counter += 3;
    // mju_sub3(residual + counter, right_box_front, right_target);
    // counter += 3;

    // sensor dim sanity check
    // TODO: use this pattern everywhere and make this a utility function
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

  // ----- Transition for quadrotor task -----
  void PNP::TransitionLocked(mjModel* model, mjData* data) {
    // set mode to GUI selection
    if (mode > 0) {
      current_mode_ = mode - 1;
    } else {
      // goal position
      const double* left_front_target = data->mocap_pos;
      const double* right_front_target = data->mocap_pos + 3;
      const double* left_back_target = data->mocap_pos + 6;
      const double* right_back_target = data->mocap_pos + 9;

      // system's position
      double* left_front_pos = SensorByName(model, data, "left_hand_front");
      double* right_front_pos = SensorByName(model, data, "right_hand_front");
      double* left_back_pos = SensorByName(model, data, "left_hand_back");
      double* right_back_pos = SensorByName(model, data, "right_hand_back");

      // position error
      double left_front_position_error[3];
      mju_sub3(left_front_position_error, left_front_pos, left_front_target);
      double left_front_position_error_norm = mju_norm3(left_front_position_error);
      double right_front_position_error[3];
      mju_sub3(right_front_position_error, right_front_pos, right_front_target);
      double right_front_position_error_norm = mju_norm3(right_front_position_error);
      double left_back_position_error[3];
      mju_sub3(left_back_position_error, left_back_pos, left_back_target);
      double left_back_position_error_norm = mju_norm3(left_back_position_error);
      double right_back_position_error[3];
      mju_sub3(right_back_position_error, right_back_pos, right_back_target);
      double right_back_position_error_norm = mju_norm3(right_back_position_error);

      if (left_front_position_error_norm <= 2.0e-1 && right_front_position_error_norm <= 2.0e-1 && left_back_position_error_norm <= 2.0e-1 && right_back_position_error_norm <= 2.0e-1) {
        // update task state
        current_mode_ += 1;
        if (current_mode_ >= 2) {
          current_mode_ = 2;
        }
      }
    }

    // set goal
    double* left_front_target_site = nullptr;
    double* right_front_target_site = nullptr;
    double* left_back_target_site = nullptr;
    double* right_back_target_site = nullptr;
    if (current_mode_ == 0) {
      left_front_target_site = SensorByName(model, data, "box1-1");
      right_front_target_site = SensorByName(model, data, "box1-2");
      left_back_target_site = SensorByName(model, data, "box1-3");
      right_back_target_site = SensorByName(model, data, "box1-4");
    }
    else if (current_mode_ == 1) {
      left_front_target_site = SensorByName(model, data, "box2-1");
      right_front_target_site = SensorByName(model, data, "box2-2");
      left_back_target_site = SensorByName(model, data, "box2-3");
      right_back_target_site = SensorByName(model, data, "box2-4");
    }
    else if (current_mode_ == 2) {
      left_front_target_site = SensorByName(model, data, "box3-1");
      right_front_target_site = SensorByName(model, data, "box3-2");
      left_back_target_site = SensorByName(model, data, "box3-3");
      right_back_target_site = SensorByName(model, data, "box3-4");
    }
    else{
      mju_error("Invalid hand reach mode: %d", current_mode_);
    }
    mju_copy3(data->mocap_pos, left_front_target_site);
    mju_copy3(data->mocap_pos + 3, right_front_target_site);
    mju_copy3(data->mocap_pos + 6, left_back_target_site);
    mju_copy3(data->mocap_pos + 9, right_back_target_site);
  }

} // namespace mjpc::h1_mani
