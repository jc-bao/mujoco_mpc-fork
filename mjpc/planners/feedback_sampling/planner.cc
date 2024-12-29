#include "mjpc/planners/feedback_sampling/planner.h"

#include <algorithm>
#include <chrono>
#include <shared_mutex>
#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include "mjpc/array_safety.h"
#include "mjpc/planners/ilqg/policy.h"
#include "mjpc/planners/ilqg/planner.h"
#include "mjpc/planners/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc
{

  namespace mju = ::mujoco::util_mjpc;

  // initialize data and settings
  void FeedbackSamplingPlanner::Initialize(mjModel *model, const Task &task)
  {
    // clear and resize data
    data_.clear();
    ResizeMjData(model, 1);

    this->model = model;
    this->task = &task;

    // noise std from model numeric
    noise_exploration[0] = GetNumberOrDefault(0.1, model, "sampling_exploration");
    int se_id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_exploration");
    if (se_id >= 0 && model->numeric_size[se_id] > 1)
    {
      int se_adr = model->numeric_adr[se_id];
      noise_exploration[1] = model->numeric_data[se_adr + 1];
    }

    // number of trajectories
    num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

    // feedback scale
    feedback_scale = GetNumberOrDefault(1.0, model, "sampling_scale");

    // spline points
    num_spline_points = GetNumberOrDefault(kMaxTrajectoryHorizon, model,
                                           "sampling_spline_points");
    interpolation_ = GetNumberOrDefault(mjpc::spline::SplineInterpolation::kCubicSpline, model,
                                        "sampling_representation");

    // number of sampling update before feedback is applied
    num_sampling_update_before_feedback = GetNumberOrDefault(3, model, "sampling_update_before_feedback");

    // cost variance threshold
    cost_variance_threshold_ = GetNumberOrDefault(100.0, model, "sampling_cost_variance_threshold");

    // feedback scale gain
    k_feedback_scale_gain = GetNumberOrDefault(1.0, model, "sampling_feedback_scale_gain");

    // improvement value gamma
    improvement_value_gamma = GetNumberOrDefault(0.95, model, "sampling_improvement_value_gamma");

    winner = 0;

    // Initialize ilqg_solver
    ilqg_solver.Initialize(model, task);
  }

  // allocate memory
  void FeedbackSamplingPlanner::Allocate()
  {
    int num_state = model->nq + model->nv + model->na;

    state.resize(num_state);
    mocap.resize(7 * model->nmocap);
    userdata.resize(model->nuserdata);

    policy.Allocate(model, *task, kMaxTrajectoryHorizon);
    previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);

    for (int i = 0; i < kMaxTrajectory; i++)
    {
      candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
      trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                               task->num_trace, kMaxTrajectoryHorizon);
      trajectory[i].Allocate(kMaxTrajectoryHorizon);
    }

    // Allocate ilqg_solver
    ilqg_solver.Allocate();
  }

  // reset memory
  void FeedbackSamplingPlanner::Reset(int horizon,
                                      const double *initial_repeated_action)
  {
    std::fill(state.begin(), state.end(), 0.0);
    std::fill(mocap.begin(), mocap.end(), 0.0);
    std::fill(userdata.begin(), userdata.end(), 0.0);
    time = 0.0;

    {
      const std::unique_lock<std::shared_mutex> lock(mtx_);
      policy.Reset(horizon, initial_repeated_action);
      previous_policy.Reset(horizon, initial_repeated_action);
    }

    for (int i = 0; i < kMaxTrajectory; i++)
    {
      candidate_policy[i].Reset(horizon, initial_repeated_action);
      trajectory[i].Reset(horizon, initial_repeated_action);
    }

    sampling_improvement = 0.0;
    improvement_value = 0.0;
    ilqr_improvement = 0.0;
    winner = 0;

    // Reset ilqg_solver
    ilqg_solver.Reset(horizon, initial_repeated_action);

    // reset update count
    update_cnt = 0;
  }

  // set state
  void FeedbackSamplingPlanner::SetState(const State &state_in)
  {
    state_in.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
                    &this->time);

    // set state in ilqg_solver
    ilqg_solver.SetState(state_in);
  }

  // run iLQG to update nominal policy
  void FeedbackSamplingPlanner::UpdateNominalPolicy(int horizon, ThreadPool &pool)
  {
    // Run iLQG optimization to get a new nominal policy
    // update every num_sampling_update_before_feedback updates
    ilqg_solver.OptimizePolicy(horizon, pool);
    ilqr_improvement = ilqg_solver.improvement;

    // Copy iLQG when its cost variance is less than the threshold
    if (ilqg_solver.cost_variance < cost_variance_threshold_)
    {
      const std::unique_lock<std::shared_mutex> lock(mtx_);
      policy.CopyFrom(ilqg_solver.policy, ilqg_solver.policy.trajectory.horizon);
      // Ensure the main policy uses full feedback
      policy.feedback_scaling = 1.0;
    }
    else
    {
      policy.feedback_scaling = 0.0;
    }
  }

  // optimize nominal policy using feedback-based sampling
  void FeedbackSamplingPlanner::OptimizePolicy(int horizon, ThreadPool &pool)
  {
    // Update nominal policy from iLQG, update ilgr in a lower frequency since it is
    if (update_cnt % num_sampling_update_before_feedback == 0)
    {
      UpdateNominalPolicy(horizon, pool);
    }

    // Optimize candidates
    OptimizePolicyCandidates(1, horizon, pool);

    // update policy with best candidate
    auto policy_update_start = std::chrono::steady_clock::now();
    CopyCandidateToPolicy(0);

    double best_return = trajectory[0].total_return;
    sampling_improvement = mju_max(best_return - trajectory[winner].total_return, 0.0);

    // update improvement value
    improvement_value = sampling_improvement + improvement_value_gamma * improvement_value;
    // clip the improvement value between 0 and 3
    improvement_value = mju_min(improvement_value, 3.0);

    policy_update_compute_time = GetDuration(policy_update_start);

    // update feedback scaling which is exp(-sampling_improvement * k_feedback_scale_gain)
    feedback_scale = exp(-improvement_value * k_feedback_scale_gain);
    // make sure feedback scaling is upper bounded by 0.9
    feedback_scale = mju_min(feedback_scale, 0.9);
    // update num_sampling_update_before_feedback
    // float feedback_scale_inv = mju_min(2.0 / feedback_scale, 100.0);
    // num_sampling_update_before_feedback = static_cast<int>(feedback_scale_inv);

    // increment update count
    update_cnt++;
  }

  // compute trajectory using nominal policy
  void FeedbackSamplingPlanner::NominalTrajectory(int horizon, ThreadPool &pool)
  {
    auto nominal_policy_fn = [this](double *action, const double *st, double t)
    {
      this->policy.Action(action, st, t);
    };

    trajectory[0].Rollout(nominal_policy_fn, task, model, data_[0].get(),
                          state.data(), time, mocap.data(), userdata.data(),
                          horizon);
  }

  // set action from policy
  void FeedbackSamplingPlanner::ActionFromPolicy(double *action, const double *st,
                                                 double t, bool use_previous)
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    if (use_previous)
    {
      previous_policy.Action(action, st, t);
    }
    else
    {
      policy.Action(action, st, t);
    }
  }

  int FeedbackSamplingPlanner::OptimizePolicyCandidates(int ncandidates, int horizon,
                                                        ThreadPool &pool)
  {
    ncandidates = std::min(ncandidates, num_trajectory_);

    ResizeMjData(model, pool.NumThreads());

    auto rollouts_start = std::chrono::steady_clock::now();
    Rollouts(num_trajectory_, horizon, pool);

    // sort candidates by score
    trajectory_order.clear();
    trajectory_order.reserve(num_trajectory_);
    for (int i = 0; i < num_trajectory_; i++)
    {
      trajectory_order.push_back(i);
    }

    std::partial_sort(
        trajectory_order.begin(), trajectory_order.begin() + ncandidates,
        trajectory_order.end(), [this](int a, int b)
        { return trajectory[a].total_return < trajectory[b].total_return; });

    rollouts_compute_time = GetDuration(rollouts_start);

    return ncandidates;
  }

  double FeedbackSamplingPlanner::CandidateScore(int candidate) const
  {
    return trajectory[trajectory_order[candidate]].total_return;
  }

  void FeedbackSamplingPlanner::ActionFromCandidatePolicy(double *action, int candidate,
                                                          const double *st, double t)
  {
    candidate_policy[trajectory_order[candidate]].Action(action, st, t);
  }

  void FeedbackSamplingPlanner::CopyCandidateToPolicy(int candidate)
  {
    winner = trajectory_order[candidate];
    {
      const std::unique_lock<std::shared_mutex> lock(mtx_);
      previous_policy = policy;
      policy.CopyFrom(candidate_policy[winner], candidate_policy[winner].trajectory.horizon);
      // Reset the feedback scaling to 1.0 for the chosen policy
      policy.feedback_scaling = 1.0;
    }
  }

  // return best trajectory
  const Trajectory *FeedbackSamplingPlanner::BestTrajectory()
  {
    return winner >= 0 ? &trajectory[winner] : nullptr;
  }

  void FeedbackSamplingPlanner::AddNoiseToPolicy(int i)
  {
    auto noise_start = std::chrono::steady_clock::now();
    absl::BitGen gen_;

    double std_dev = noise_exploration[0];
    constexpr double kStd2Proportion = 0.2;
    if (noise_exploration[1] > 0 && absl::Bernoulli(gen_, kStd2Proportion))
    {
      std_dev = noise_exploration[1];
    }

    iLQGPolicy &cp = candidate_policy[i];
    int horizon = cp.trajectory.horizon;
    int nu = model->nu;

    // Create a local spline for noise
    mjpc::spline::TimeSpline local_noise_spline(nu, interpolation_);
    local_noise_spline.Reserve(num_spline_points);

    // Add nodes to local spline
    for (int idx = 0; idx < num_spline_points; idx++)
    {
      double spline_time = (num_spline_points == 1)
                               ? 0.0
                               : double(idx) * (horizon - 1) / (num_spline_points - 1);
      std::vector<double> node_noise(nu, 0.0);
      for (int u = 0; u < nu; u++)
      {
        double scale = 0.5 * (model->actuator_ctrlrange[2 * u + 1] - model->actuator_ctrlrange[2 * u]);
        node_noise[u] = absl::Gaussian<double>(gen_, 0.0, scale * std_dev);
      }
      local_noise_spline.AddNode(spline_time, absl::MakeConstSpan(node_noise));
    }

    // Sample from local spline
    std::vector<double> noise_t(nu);
    for (int t = 0; t < horizon - 1; t++)
    {
      local_noise_spline.Sample(static_cast<double>(t), absl::MakeSpan(noise_t));

      double *action_t = &cp.trajectory.actions[t * nu];
      for (int u = 0; u < nu; u++)
      {
        action_t[u] += noise_t[u];
      }

      // Clamp the action after adding noise
      Clamp(action_t, model->actuator_ctrlrange, nu);
    }

    noise_compute_time = GetDuration(noise_start);
  }

  void FeedbackSamplingPlanner::Rollouts(int num_trajectory, int horizon, ThreadPool &pool)
  {
    noise_compute_time = 0.0;
    int count_before = pool.GetCount();

    for (int i = 0; i < num_trajectory; i++)
    {
      pool.Schedule([this, horizon, i]()
                    {
      {
        const std::shared_lock<std::shared_mutex> lock(this->mtx_);
        candidate_policy[i].CopyFrom(this->policy, this->policy.trajectory.horizon);
        // Set feedback scaling for candidate policies (only for sampling)
        candidate_policy[i].feedback_scaling = this->feedback_scale;
        // sampling token
        absl::BitGen gen_;
        // randomly select .0.25 * num_trajectory to disable feedback
        if (absl::Bernoulli(gen_, 0.25))
        {
          candidate_policy[i].feedback_scaling = 0.0;
        }
      }

      if (i != 0) {
        AddNoiseToPolicy(i);
      }

      // rollout candidate policy i
      auto candidate_pi = [this, i](double* action, const double* st, double t) {
        candidate_policy[i].Action(action, st, t);
      };

      trajectory[i].Rollout(candidate_pi, task, model,
                            data_[ThreadPool::WorkerId()].get(), state.data(),
                            time, mocap.data(), userdata.data(), horizon); });
    }

    pool.WaitCount(count_before + num_trajectory);
    pool.ResetCount();
  }

  // visualize planner-specific traces
  void FeedbackSamplingPlanner::Traces(mjvScene *scn)
  {
    // sample color
    float color[4];
    color[0] = 1.0;
    color[1] = 1.0;
    color[2] = 1.0;
    color[3] = 1.0;

    // width of a sample trace, in pixels
    double width = GetNumberOrDefault(3, model, "agent_sample_width");

    // scratch
    double zero3[3] = {0};
    double zero9[9] = {0};

    // best
    auto best = this->BestTrajectory();

    // sample traces
    for (int k = 0; k < num_trajectory_; k++)
    {
      // skip winner
      if (k == winner)
        continue;

      // plot sample
      for (int i = 0; i < best->horizon - 1; i++)
      {
        if (scn->ngeom + task->num_trace > scn->maxgeom)
          break;
        for (int j = 0; j < task->num_trace; j++)
        {
          // initialize geometry
          mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                       color);

          // make geometry
          mjv_connector(
              &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
              trajectory[k].trace.data() + 3 * task->num_trace * i + 3 * j,
              trajectory[k].trace.data() + 3 * task->num_trace * (i + 1) + 3 * j);

          // increment number of geometries
          scn->ngeom += 1;
        }
      }
    }
  }

  void FeedbackSamplingPlanner::GUI(mjUI &ui)
  {
    mjuiDef defFeedbackSampling[] = {
        {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "1 36"},
        {mjITEM_SLIDERNUM, "Noise Std", 2, noise_exploration, "0 1"},
        {mjITEM_SLIDERNUM, "Noise Std2", 2, noise_exploration + 1, "0 1"},
        // Add a GUI element to control feedback_scale if desired
        {mjITEM_SLIDERNUM, "Feedback Scale", 2, &feedback_scale, "0 1"},
        {mjITEM_SLIDERINT, "Spline Points", 2, &num_spline_points, "1 10"},
        {mjITEM_SELECT, "Spline", 2, &interpolation_, "Zero\nLinear\nCubic"},
        {mjITEM_SLIDERINT, "Sampling Update Before Feedback", 10, &num_sampling_update_before_feedback, "1 50"},
        {mjITEM_SLIDERNUM, "Cost Variance Threshold", 2, &cost_variance_threshold_, "0 100"},
        {mjITEM_SLIDERNUM, "Feedback Scale Gain", 1, &k_feedback_scale_gain, "0.1 10"},
        {mjITEM_SLIDERNUM, "Improvement Value Gamma", 1, &improvement_value_gamma, "0.1 1.0"},
        {mjITEM_END}};

    mjui_add(&ui, defFeedbackSampling);
  }

  void FeedbackSamplingPlanner::Plots(mjvFigure *fig_planner, mjvFigure *fig_timer,
                                      int planner_shift, int timer_shift,
                                      int planning, int *shift)
  {
    double planner_bounds[2] = {-6.0, 6.0};

    // improvement plot
    PlotUpdateData(fig_planner, planner_bounds,
                   fig_planner->linedata[0 + planner_shift][0] + 1,
                   mju_log10(mju_max(sampling_improvement, 1.0e-6)),
                   100, 0 + planner_shift, 0, 1, -100);
    mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Sampling Improvement");

    // ilqr improvement plot
    PlotUpdateData(fig_planner, planner_bounds,
                   fig_planner->linedata[1 + planner_shift][0] + 1,
                   mju_log10(mju_max(ilqr_improvement, 1.0e-6)),
                   //  ilqr_improvement,
                   100, 1 + planner_shift, 0, 1, -100);
    mju::strcpy_arr(fig_planner->linename[1 + planner_shift], "iLQG Improvement");

    fig_planner->range[1][0] = planner_bounds[0];
    fig_planner->range[1][1] = planner_bounds[1];

    double timer_bounds[2] = {0.0, 1.0};

    // noise time
    PlotUpdateData(fig_timer, timer_bounds,
                   fig_timer->linedata[0 + timer_shift][0] + 1,
                   1.0e-3 * noise_compute_time * planning,
                   100, 0 + timer_shift, 0, 1, -100);
    mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise");

    // rollouts time
    PlotUpdateData(fig_timer, timer_bounds,
                   fig_timer->linedata[1 + timer_shift][0] + 1,
                   1.0e-3 * rollouts_compute_time * planning,
                   100, 1 + timer_shift, 0, 1, -100);
    mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout");

    // policy update time
    PlotUpdateData(fig_timer, timer_bounds,
                   fig_timer->linedata[2 + timer_shift][0] + 1,
                   1.0e-3 * policy_update_compute_time * planning,
                   100, 2 + timer_shift, 0, 1, -100);
    mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update");

    fig_timer->range[0][0] = -100;
    fig_timer->range[0][1] = 0;
    fig_timer->range[1][0] = 0.0;
    fig_timer->range[1][1] = 1.0;

    shift[0] += 1;
    shift[1] += 3;
  }

} // namespace mjpc
