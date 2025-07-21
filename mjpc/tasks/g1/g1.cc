#include "mjpc/tasks/g1/g1.h"

#include <cmath>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string G1::XmlPath() const {
  return GetModelPath("g1/task.xml");
}

std::string G1::Name() const { 
  return "G1"; 
}

std::vector<std::vector<float>> G1::LoadBinaryArray(const std::string& filename, int rows, int cols) {
  std::string full_path = GetModelPath("g1/" + filename);
  std::ifstream file(full_path, std::ios::binary);
  
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << full_path << std::endl;
    return std::vector<std::vector<float>>(rows, std::vector<float>(cols, 0.0f));
  }
  
  // Read all data
  std::vector<float> buffer;
  file.seekg(0, std::ios::end);
  size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  
  buffer.resize(fileSize / sizeof(float));
  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
  file.close();
  
  // Verify we have the expected amount of data
  if (buffer.size() != rows * cols) {
    std::cerr << "Warning: Expected " << rows * cols << " floats, got " << buffer.size() << std::endl;
  }
  
  // Reshape to (rows, cols)
  std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
  
  for (int i = 0; i < rows && i * cols < buffer.size(); ++i) {
    for (int j = 0; j < cols && i * cols + j < buffer.size(); ++j) {
      result[i][j] = buffer[i * cols + j];
    }
  }
  
  return result;
}

void G1::LoadTrajectoryData() {
  qpos_trajectory_ = LoadBinaryArray("CMU_01_01_poses_qpos.bin", 914, 30);
  qvel_trajectory_ = LoadBinaryArray("CMU_01_01_poses_qvel.bin", 914, 29);
  
  // Assume 30 FPS motion capture data (typical for CMU mocap)
  trajectory_dt_ = 1.0 / 50.0;  // ~0.02 seconds per frame
  
  std::cout << "Loaded trajectory data: qpos(" << qpos_trajectory_.size() 
            << ", " << (qpos_trajectory_.empty() ? 0 : qpos_trajectory_[0].size()) 
            << "), qvel(" << qvel_trajectory_.size() 
            << ", " << (qvel_trajectory_.empty() ? 0 : qvel_trajectory_[0].size()) << ")" << std::endl;
}

void G1::ResidualFn::Residual(const mjModel* model, const mjData* data,
                              double* residual) const {
  const G1* g1_task = static_cast<const G1*>(task_);
  
  // Calculate trajectory index based on current time
  int trajectory_index = static_cast<int>(data->time / g1_task->trajectory_dt_);
  
  // Clamp to valid range
  trajectory_index = std::max(0, std::min(trajectory_index, 
                                          static_cast<int>(g1_task->qpos_trajectory_.size()) - 1));
  
  // Get target positions from trajectory
  const auto& qpos_target = g1_task->qpos_trajectory_[trajectory_index];
  const auto& qvel_target = g1_task->qvel_trajectory_[trajectory_index];
  
  // Convert float trajectory data to double for qpos_tar
  std::vector<double> qpos_tar(qpos_target.begin(), qpos_target.end());
  
  // Compute difference between qpos and qpos_tar with mujoco's function mj_differentiatePos
  // This computes the velocity needed to go from current qpos to target qpos in unit time
  mj_differentiatePos(model, residual, 1.0, data->qpos, qpos_tar.data());

  // Compute simple difference between current qvel and target qvel
  // Store qvel differences after qpos differences in residual array
  for (int i = 0; i < model->nv && i < qvel_target.size(); i++) {
    residual[model->nv + i] = data->qvel[i] - static_cast<double>(qvel_target[i]);
  }
  
  // Handle case where qvel has more elements than target (pad with zeros)
  for (int i = qvel_target.size(); i < model->nv; i++) {
    residual[model->nv + i] = data->qvel[i];
  }
}

}  // namespace mjpc