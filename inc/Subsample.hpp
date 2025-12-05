#pragma once

#include "InverseIterative.hpp"
#include "PointCloud.hpp"
#include "Utils.hpp"

inline PointCloud subsample(const PointCloud &cloud, double distance,
                            float sigmaP = 1.0f, float sigmaN = 1.0f,
                            float eps = 0.5) {

  auto sampled_positions = std::vector<PointCloud::Position>{};
  auto sampled_normals = std::vector<PointCloud::Normal>{};
  PointKDTree sampled = PointKDTree<true>(sampled_positions);

  // Avoid reallocating a bunch of memory
  auto mem = PreallocatedMemory(256);

  for (size_t i = 0; i < cloud.positions.size(); ++i) {

    glm::vec3 p = cloud.positions[i];
    glm::vec3 n = cloud.normals[i];
    auto x = util::concat(p / sigmaP, n / sigmaN);

    // we get sorted neighbours
    auto neighbors = sampled.neighboursInRadius(p, distance);

    double s = 1.0;
    if (!neighbors.empty()) {
      std::vector<util::PointNormal> neighborPointNormals;
      for (auto &[idx, _] : neighbors) {
        auto &np = sampled_positions[idx];
        auto &nn = sampled_normals[idx];
        neighborPointNormals.push_back(util::concat(np / sigmaP, nn / sigmaN));
      }

      s = takeInverseIterative(x, neighborPointNormals, mem, rbfKernel, eps).s;
    }
    if (s > eps) {
      sampled_positions.push_back(p);
      sampled_normals.push_back(n);
      const auto newIndex = sampled_positions.size() - 1;
      sampled.addPoints(newIndex, newIndex);
    }
  }

  std::cout << "Reduction = "
            << 1.0f - sampled_positions.size() /
                          static_cast<float>(cloud.positions.size())
            << std::endl;

  return PointCloud(sampled_positions, sampled_normals);
}
