#ifndef MJPC_PLANNERS_FEEDBACK_SAMPLING_PLANNER_H_
#define MJPC_PLANNERS_FEEDBACK_SAMPLING_PLANNER_H_

#include <mujoco/mujoco.h>
#include <shared_mutex>
#include <vector>

#include "mjpc/planners/ilqg/planner.h"
#include "mjpc/planners/ilqg/policy.h"
#include "mjpc/planners/planner.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/spline/spline.h"
namespace mjpc
{

  // FeedbackSamplingPlanner: uses an iLQG-based feedback policy as the nominal
  // policy and samples around it to find improvements.
  class FeedbackSamplingPlanner : public RankedPlanner
  {
  public:
    // constructor
    FeedbackSamplingPlanner() = default;

    // destructor
    ~FeedbackSamplingPlanner() override = default;

    // initialize data and settings
    void Initialize(mjModel *model, const Task &task) override;

    // allocate memory
    void Allocate() override;

    // reset memory to zeros
    void Reset(int horizon,
               const double *initial_repeated_action = nullptr) override;

    // set state
    void SetState(const State &state) override;

    // optimize nominal policy using feedback-based sampling
    void OptimizePolicy(int horizon, ThreadPool &pool) override;

    // compute trajectory using nominal policy
    void NominalTrajectory(int horizon, ThreadPool &pool) override;

    // set action from policy
    void ActionFromPolicy(double *action, const double *state,
                          double time, bool use_previous = false) override;

    // generate and optimize candidates
    int OptimizePolicyCandidates(int ncandidates, int horizon,
                                 ThreadPool &pool) override;

    double CandidateScore(int candidate) const override;
    void ActionFromCandidatePolicy(double *action, int candidate,
                                   const double *state, double time) override;
    void CopyCandidateToPolicy(int candidate) override;

    // return best trajectory
    const Trajectory *BestTrajectory() override;

    // visualize planner-specific traces
    void Traces(mjvScene *scn) override;

    // planner-specific GUI elements
    void GUI(mjUI &ui) override;

    // planner-specific plots
    void Plots(mjvFigure *fig_planner, mjvFigure *fig_timer, int planner_shift,
               int timer_shift, int planning, int *shift) override;

    // return number of parameters
    int NumParameters() override
    {
      return policy.trajectory.horizon * model->nu;
    }

  private:
    // add noise to the candidate policy
    void AddNoiseToPolicy(int i);

    // rollout candidates
    void Rollouts(int num_trajectory, int horizon, ThreadPool &pool);

    // update nominal policy via iLQG
    void UpdateNominalPolicy(int horizon, ThreadPool &pool);

    // members
    mjModel *model = nullptr;
    const Task *task = nullptr;

    std::vector<double> state;
    double time = 0.0;
    std::vector<double> mocap;
    std::vector<double> userdata;

    // iLQG solver instance to produce nominal feedback policy
    iLQGPlanner ilqg_solver;

    // policies
    iLQGPolicy policy;
    iLQGPolicy previous_policy;
    iLQGPolicy candidate_policy[kMaxTrajectory];

    Trajectory trajectory[kMaxTrajectory];
    std::vector<int> trajectory_order;

    // noise std dev
    double noise_exploration[2] = {0.1, 0.0};

    int winner = -1;
    double sampling_improvement = 0.0;
    double ilqr_improvement = 0.0;

    // number of trajectories
    int num_trajectory_ = 10;

    // controller feedback scale
    double feedback_scale = 1.0;

    // number of sampling update before feedback is applied
    int num_sampling_update_before_feedback = 3;
    int update_cnt = 0;

    // spline
    int num_spline_points = 10;
    mjpc::spline::SplineInterpolation interpolation_ =
        mjpc::spline::SplineInterpolation::kZeroSpline;

    // timing
    double noise_compute_time = 0.0;
    double rollouts_compute_time = 0.0;
    double policy_update_compute_time = 0.0;

    // cost variance threshold
    double cost_variance_threshold_ = 10.0;

    // feedback scale gain
    double k_feedback_scale_gain = 1.0;

    // improvement value gamma
    double improvement_value_gamma = 0.95;

    // improvement value 
    double improvement_value = 0.0;

    mutable std::shared_mutex mtx_;
  };

} // namespace mjpc

#endif // MJPC_PLANNERS_FEEDBACK_SAMPLING_PLANNER_H_