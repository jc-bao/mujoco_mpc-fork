#include "mjpc/tasks/g1/tracking_realtime/tracking_realtime.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <tuple>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace
{
    // compute interpolation between mocap frames
    std::tuple<int, int, double, double> ComputeInterpolationValues(double index,
                                                                    int max_index)
    {
        int index_0 = std::floor(std::clamp(index, 0.0, (double)max_index));
        int index_1 = std::min(index_0 + 1, max_index);

        double weight_1 = std::clamp(index, 0.0, (double)max_index) - index_0;
        double weight_0 = 1.0 - weight_1;

        return {index_0, index_1, weight_0, weight_1};
    }

    // Hardcoded constant matching keyframes from CMU mocap dataset.
    constexpr double kFps = 30.0;

    // return length of motion trajectory (Now hardcoded to 50)
    int MotionLength(int id) { return 50; }

    // return starting keyframe index for motion (Now hardcoded to 0)
    int MotionStartIndex(int id)
    {
        return 0;
    }

    // names for g1 bodies
    const std::array<std::string, 16> body_names = {
        "pelvis",
        "head",
        "ltoe",
        "rtoe",
        "lheel",
        "rheel",
        "lknee",
        "rknee",
        "lhand",
        "rhand",
        "lelbow",
        "relbow",
        "lshoulder",
        "rshoulder",
        "lhip",
        "rhip",
    };

    // hardcoded default mocap positions
    const std::array<double, 3 * 16> default_mocap_pos = {
        -0.08679, 0.19292, 0.86056, -0.04738, 0.30140, 1.36626, -0.17402, 0.34120, -0.00107, 0.00010, 0.29688, 0.00416, -0.16921, 0.17126, -0.00093, -0.03162, 0.12988, 0.00260, -0.18658, 0.21238, 0.39614, 0.01303, 0.19736, 0.39839, -0.29056, 0.32638, 0.73196, 0.12556, 0.30252, 0.71085, -0.25784, 0.25329, 0.89793, 0.10067, 0.22799, 0.87754, -0.16856, 0.24912, 1.07646, 0.03103, 0.23479, 1.06491, -0.17502, 0.20443, 0.70755, -0.04873, 0.17857, 0.70803
    };

} // namespace

namespace mjpc::g1
{

    std::string TrackingRealtime::XmlPath() const
    {
        return GetModelPath("g1/tracking_realtime/task.xml");
    }
    std::string TrackingRealtime::Name() const { return "G1 Tracking Realtime"; }

    // ------------- Residuals for humanoid tracking_realtime task -------------
    //   Number of residuals:
    //     Residual (0): Joint vel: minimise joint velocity
    //     Residual (1): Control: minimise control
    //     Residual (2-11): TrackingRealtime position: minimise tracking_realtime position error
    //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
    //     Residual (11-20): TrackingRealtime velocity: minimise tracking_realtime velocity error
    //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
    //   Number of parameters: 0    
    // ----------------------------------------------------------------
    void TrackingRealtime::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                                double *residual) const
    {
        // ----- get mocap frames ----- //
        // get motion start index
        int start = MotionStartIndex(current_mode_);
        // get motion trajectory length
        int length = MotionLength(current_mode_);
        double reference_time = data->userdata[0];
        double current_index = (data->time - reference_time) * kFps + start;
        int last_key_index = start + length - 1;
        // make sure current_index is within the range of the motion
        current_index = std::clamp(current_index, 0.0, (double)last_key_index);

        // Positions:
        // We interpolate linearly between two consecutive key frames in order to
        // provide smoother signal for tracking_realtime.
        int key_index_0, key_index_1;
        double weight_0, weight_1;
        std::tie(key_index_0, key_index_1, weight_0, weight_1) =
            ComputeInterpolationValues(current_index, last_key_index);

        // ----- residual ----- //
        int counter = 0;

        // ----- joint velocity ----- //
        mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
        counter += model->nv - 6;

        // ----- action ----- //
        mju_copy(&residual[counter], data->ctrl, model->nu);
        counter += model->nu;

        // ----- position ----- //
        // Compute interpolated frame.
        auto get_body_mpos = [&](const std::string &body_name, double result[3])
        {
            // this frame
            std::string mocap_body_name = "mocap[" + body_name + "]";
            int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
            assert(0 <= mocap_body_id);
            int body_mocapid = model->body_mocapid[mocap_body_id];
            assert(0 <= body_mocapid);

            // next frame
            // std::string mocap_body_name_next = "mocap[" + body_name + "]";
            // int mocap_body_id_next = mj_name2id(model, mjOBJ_BODY, mocap_body_name_next.c_str());
            // assert(0 <= mocap_body_id_next);
            // int body_mocapid_next = model->body_mocapid[mocap_body_id_next];
            // assert(0 <= body_mocapid_next);

            // current frame
            mju_scl3(
                result,
                data->userdata + model->nmocap * 3 * key_index_0 + 3 * body_mocapid + 1, // +1 because of time
                weight_0);

            // next frame
            mju_addToScl3(
                result,
                data->userdata + model->nmocap * 3 * key_index_1 + 3 * body_mocapid + 1, // +1 because of time
                weight_1);
        };

        auto get_body_sensor_pos = [&](const std::string &body_name,
                                       double result[3])
        {
            std::string pos_sensor_name = "tracking_pos[" + body_name + "]";
            double *sensor_pos = SensorByName(model, data, pos_sensor_name.c_str());
            mju_copy3(result, sensor_pos);
        };

        // compute marker and sensor averages
        double avg_mpos[3] = {0};
        double avg_sensor_pos[3] = {0};
        int num_body = 0;
        for (const auto &body_name : body_names)
        {
            double body_mpos[3];
            double body_sensor_pos[3];
            get_body_mpos(body_name, body_mpos);
            mju_addTo3(avg_mpos, body_mpos);
            get_body_sensor_pos(body_name, body_sensor_pos);
            mju_addTo3(avg_sensor_pos, body_sensor_pos);
            num_body++;
        }
        mju_scl3(avg_mpos, avg_mpos, 1.0 / num_body);
        mju_scl3(avg_sensor_pos, avg_sensor_pos, 1.0 / num_body);

        // residual for averages
        mju_sub3(&residual[counter], avg_mpos, avg_sensor_pos);
        counter += 3;

        for (const auto &body_name : body_names)
        {
            double body_mpos[3];
            get_body_mpos(body_name, body_mpos);

            // current position
            double body_sensor_pos[3];
            get_body_sensor_pos(body_name, body_sensor_pos);

            mju_subFrom3(body_mpos, avg_mpos);
            mju_subFrom3(body_sensor_pos, avg_sensor_pos);

            mju_sub3(&residual[counter], body_mpos, body_sensor_pos);

            counter += 3;
        }

        // ----- velocity ----- //
        for (const auto &body_name : body_names)
        {
            std::string mocap_body_name = "mocap[" + body_name + "]";
            std::string linvel_sensor_name = "tracking_linvel[" + body_name + "]";
            int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
            assert(0 <= mocap_body_id);
            int body_mocapid = model->body_mocapid[mocap_body_id];
            assert(0 <= body_mocapid);

            // std::string mocap_body_name_next = "mocap[" + body_name + "]" + std::to_string(current_index + 1);
            // int mocap_body_id_next = mj_name2id(model, mjOBJ_BODY, mocap_body_name_next.c_str());
            // assert(0 <= mocap_body_id_next);
            // int body_mocapid_next = model->body_mocapid[mocap_body_id_next];
            // assert(0 <= body_mocapid_next);

            // compute finite-difference velocity
            // mju_copy3(
            //     &residual[counter],
            //     model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid);
            // mju_subFrom3(
            //     &residual[counter],
            //     model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid);
            mju_copy3(
                &residual[counter],
                data->userdata + model->nmocap * 3 * key_index_1 + 3 * body_mocapid + 1);
            mju_subFrom3(
                &residual[counter],
                data->userdata + model->nmocap * 3 * key_index_0 + 3 * body_mocapid + 1);
            mju_scl3(&residual[counter], &residual[counter], kFps);

            // subtract current velocity
            double *sensor_linvel =
                SensorByName(model, data, linvel_sensor_name.c_str());
            mju_subFrom3(&residual[counter], sensor_linvel);

            counter += 3;
        }

        CheckSensorDim(model, counter);
    }

    // --------------------- Transition for G1 task -------------------------
    //   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
    //   Linearly interpolate between two consecutive key frames in order to
    //   smooth the transitions between keyframes.
    // ----------------------------------------------------------------------------
    void TrackingRealtime::TransitionLocked(mjModel *model, mjData *d)
    {
        // get motion start index
        int start = MotionStartIndex(mode);
        // get motion trajectory length
        int length = MotionLength(mode);

        // check for motion switch
        if (residual_.current_mode_ != mode || d->time == 0.0)
        {
            residual_.current_mode_ = mode;             // set motion id
            residual_.reference_time_ = d->userdata[0]; // set reference time

            // set initial state
            mju_copy(d->qpos, model->key_qpos + model->nq * 0, model->nq);
            mju_copy(d->qvel, model->key_qvel + model->nv * 0, model->nv);

            // if mode is 0, set userdata to default mocap pos
            if (mode == 0)
            {
                // mju_copy(d->mocap_pos, model->key_mpos, model->nmocap * 3);
                for (int i = 0; i < model->nmocap * 3 * 50; i++)
                {
                    d->userdata[i + 1] = default_mocap_pos[i % (3 * 16)];
                }
                d->userdata[0] = d->time;
            }
        }


        // indices
        double current_index = (d->time - residual_.reference_time_) * kFps + start;
        int last_key_index = start + length - 1;
        current_index = std::clamp(current_index, 0.0, (double)last_key_index);
        // Positions:
        // We interpolate linearly between two consecutive key frames in order to
        // provide smoother signal for tracking.
        int key_index_0, key_index_1;
        double weight_0, weight_1;
        std::tie(key_index_0, key_index_1, weight_0, weight_1) =
            ComputeInterpolationValues(current_index, last_key_index);

        mj_markStack(d);

        mjtNum *mocap_pos_0 = mj_stackAllocNum(d, 3 * model->nmocap);
        mjtNum *mocap_pos_1 = mj_stackAllocNum(d, 3 * model->nmocap);

        // Compute interpolated frame.
        // mju_scl(mocap_pos_0, model->key_mpos + model->nmocap * 3 * key_index_0,
        //         weight_0, model->nmocap * 3);

        // mju_scl(mocap_pos_1, model->key_mpos + model->nmocap * 3 * key_index_1,
        //         weight_1, model->nmocap * 3);
        mju_scl(mocap_pos_0, d->userdata + model->nmocap * 3 * key_index_0 + 1,
                weight_0, model->nmocap * 3);

        mju_scl(mocap_pos_1, d->userdata + model->nmocap * 3 * key_index_1 + 1,
                weight_1, model->nmocap * 3);

        mju_copy(d->mocap_pos, mocap_pos_0, model->nmocap * 3);
        mju_addTo(d->mocap_pos, mocap_pos_1, model->nmocap * 3);

        mj_freeStack(d);
    }

} // namespace mjpc::g1
