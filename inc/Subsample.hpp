#pragma once

#include "InverseIterative.hpp"
#include "PointCloud.hpp"
#include "Utils.hpp"

inline PointCloud poissonDiskSubsample(const PointCloud &cloud, double distance,
                                       float sigmaP = 1.0f, float sigmaN = 1.0f,
                                       float eps = 0.5) {
  // empty pointcloud
  PointCloud sampled;

  // Avoid reallocating a bunch of memory
  auto mem = PreallocatedMemory(256);

  for (size_t i = 0; i < cloud.positions.size(); ++i) {

    glm::vec3 p = cloud.positions[i];
    glm::vec3 n = cloud.normals[i];
    auto x = util::concat(p / sigmaP, n / sigmaN);

    // we get sorted neighbours
    auto neighbors = sampled.tree->neighboursInRadius(p, distance);

    double s = 1.0;
    if (!neighbors.empty()) {
      std::vector<util::PointNormal> neighborPointNormals;
      for (auto &[idx, _] : neighbors) {
        auto &np = sampled.positions[idx];
        auto &nn = sampled.normals[idx];
        neighborPointNormals.push_back(util::concat(np / sigmaP, nn / sigmaN));
      }

      s = takeInverseIterative(x, neighborPointNormals, mem, rbfKernel, eps).s;
    }
    if (s > eps) {
      sampled.insertPoint(p, n);
    }
  }

  std::cout << "Reduction = "
            << 1.0f - sampled.positions.size() /
                          static_cast<float>(cloud.positions.size())
            << std::endl;

  return sampled;
}
