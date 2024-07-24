// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/spot/spot.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc
{
    std::string Spot::XmlPath() const
    {
        return GetModelPath("spot/task.xml");
    }
    std::string Spot::Name() const { return "Spot"; }

    // ---------- Residuals for spot manipulation task ---------
    //   Number of residuals: 1
    //     Residual (0): hand - target
    // ------------------------------------------------------------
    void Spot::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                    double *residual) const
    {
        int counter = 0;

        // pos
        double *hand_pos = SensorByName(model, data, "hand_pos");
        double *box_pos = SensorByName(model, data, "target_pos");
        mju_sub3(residual + counter, hand_pos, box_pos);
        counter += 3;

        // orientation
        double *hand_quat = SensorByName(model, data, "hand_quat");
        double *box_quat = SensorByName(model, data, "target_quat");
        mju_subQuat(residual + counter, hand_quat, box_quat);
        counter += 3;

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
} // namespace mjpc
