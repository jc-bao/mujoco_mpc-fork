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

#include "mjpc/tasks/tasks.h"

#include <memory>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/tasks/acrobot/acrobot.h"
#include "mjpc/tasks/allegro/allegro.h"
#include "mjpc/tasks/bimanual/insert/insert.h"
#include "mjpc/tasks/bimanual/handover/handover.h"
#include "mjpc/tasks/bimanual/reorient/reorient.h"
#include "mjpc/tasks/cartpole/cartpole.h"
#include "mjpc/tasks/fingers/fingers.h"
#include "mjpc/tasks/humanoid/interact/interact.h"
#include "mjpc/tasks/humanoid/stand/stand.h"
#include "mjpc/tasks/humanoid/tracking/tracking.h"
#include "mjpc/tasks/humanoid/walk/walk.h"
#include "mjpc/tasks/manipulation/manipulation.h"
// DEEPMIND INTERNAL IMPORT
#include "mjpc/tasks/op3/stand.h"
#include "mjpc/tasks/panda/panda.h"
#include "mjpc/tasks/particle/particle.h"
#include "mjpc/tasks/quadrotor/quadrotor.h"
#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/rubik/solve.h"
#include "mjpc/tasks/shadow_reorient/hand.h"
#include "mjpc/tasks/swimmer/swimmer.h"
#include "mjpc/tasks/walker/walker.h"
// Chaoyi
#include "mjpc/tasks/spot/spot.h"
#include "mjpc/tasks/spot2/spot2.h"
#include "mjpc/tasks/go2/go2.h"
#include "mjpc/tasks/gr1/gr1.h"

namespace mjpc
{

    std::vector<std::shared_ptr<Task>> GetTasks()
    {
        return {
            std::make_shared<Acrobot>(),
            std::make_shared<Allegro>(),
            std::make_shared<aloha::Handover>(),
            std::make_shared<aloha::Insert>(),
            std::make_shared<aloha::Reorient>(),
            std::make_shared<Cartpole>(),
            std::make_shared<Fingers>(),
            std::make_shared<humanoid::Interact>(),
            std::make_shared<humanoid::Stand>(),
            std::make_shared<humanoid::Tracking>(),
            std::make_shared<humanoid::Walk>(),
            std::make_shared<manipulation::Bring>(),
            // DEEPMIND INTERNAL TASKS
            std::make_shared<OP3>(),
            std::make_shared<Panda>(),
            std::make_shared<Particle>(),
            std::make_shared<ParticleFixed>(),
            std::make_shared<Rubik>(),
            std::make_shared<ShadowReorient>(),
            std::make_shared<Quadrotor>(),
            std::make_shared<QuadrupedFlat>(),
            std::make_shared<QuadrupedHill>(),
            std::make_shared<Swimmer>(),
            std::make_shared<Walker>(),
            // Chaoyi
            std::make_shared<Spot>(),
            std::make_shared<Spot2Flat>(),
            std::make_shared<Go2Flat>(),
            std::make_shared<Go2Hill>(),
            std::make_shared<GR1Flat>(),
            std::make_shared<GR1Hill>(),
        };
    }
} // namespace mjpc
