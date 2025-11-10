#pragma once

#include "InverseIterative.hpp"
#include "PointCloud.hpp"

inline auto poissonDiskSubsample(const PointCloud &cloud, double distance,
                                 double threshold = 0.5) {
  // empty pointcloud
  PointCloud sampled;

  for (size_t i = 0; i < cloud.positions.size(); ++i) {

    glm::vec3 p = cloud.positions[i]; // / static_cast<float>(sigmaP);
    glm::vec3 n = cloud.normals[i];   // / static_cast<float>(sigmaN);

    // we get sorted neighbours
    auto neighbors = sampled.neighboursInRadius(p, distance);

    std::vector<size_t> neighborIndices;
    for (auto &[idx, _] : neighbors)
      neighborIndices.push_back(idx);

    double reconErr = 1.0;
    if (!neighborIndices.empty()) {
      reconErr =
          1.0 - takeInverseIterative(p, n, sampled, neighborIndices, threshold);
    }

    if (reconErr > threshold) {
      sampled.insertPoint(p, n);
    }
  }

  return sampled;
}
