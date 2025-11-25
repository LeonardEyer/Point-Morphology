#pragma once

#include "InverseIterative.hpp"
#include "PointCloud.hpp"

inline PointCloud poissonDiskSubsample(const PointCloud &cloud, double distance,
                                       float sigmaP = 1.0f, float sigmaN = 1.0f,
                                       float eps = 0.5) {
  // empty pointcloud
  PointCloud sampled;

  for (size_t i = 0; i < cloud.positions.size(); ++i) {

    glm::vec3 p = cloud.positions[i] / sigmaP;
    glm::vec3 n = cloud.normals[i] / sigmaN;

    // we get sorted neighbours
    auto neighbors = sampled.tree->neighboursInRadius(p, distance);

    std::vector<size_t> neighborIndices;
    for (auto &[idx, _] : neighbors)
      neighborIndices.push_back(idx);

    double s = 1.0;
    if (!neighborIndices.empty()) {
      s = 1.0 - takeInverseIterative(p, n, sampled, neighborIndices, eps).s;
    }

    if (s > eps) {
      sampled.insertPoint(p, n);
    }
  }

  for (auto i = 0; i < sampled.positions.size(); i++) {
    sampled.positions[i] *= sigmaP;
    sampled.normals[i] *= sigmaN;
  }

  std::cout << "Reduction = "
            << 1.0f - sampled.positions.size() /
                          static_cast<float>(cloud.positions.size())
            << std::endl;

  return sampled;
}
