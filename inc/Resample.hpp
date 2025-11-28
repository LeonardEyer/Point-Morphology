#pragma once

#include "APSS.hpp"
#include "InverseIterative.hpp"
#include "PointCloud.hpp"
#include "Utils.hpp"

inline PointCloud resample(const PointCloud &cloud, double gaussianStd,
                           float sigmaP, float sigmaN, size_t iterations = 10) {

  constexpr auto sigma = 1.0f;

  auto resampled = PointCloud(cloud.positions, cloud.normals);
  const auto apss = APSS(cloud);
  const auto nPoints = resampled.positions.size();

  auto mem = PreallocatedMemory(256);

  for (auto iter = 0; iter < iterations; ++iter) {
    for (auto i = 0; i < nPoints; ++i) {

      if (i != 0 && i % 1000 == 0) {
        std::cout << "Progress = "
                  << (iter * nPoints + i + 1) /
                         static_cast<float>(iterations * nPoints)
                  << std::endl;
      }

      const auto &p = resampled.positions[i];
      const auto &n = resampled.normals[i];
      const auto x = util::concat(p / sigmaP, n / sigmaN);

      auto neighbours = resampled.tree->neighboursInRadius(p, gaussianStd);

      if (neighbours.empty()) {
        // std::cout << "No neighbours. skipping" << std::endl;
        continue;
      }

      std::vector<util::PointNormal> neighbourPointNormals;
      for (auto &[idx, _] : neighbours) {
        neighbourPointNormals.push_back(
            util::concat(resampled.positions[idx] / sigmaP,
                         resampled.normals[idx] / sigmaN));
      }

      const auto res =
          takeInverseIterative(x, neighbourPointNormals, mem, rbfKernel, -1);

      const auto grad =
          importance_grad(x, neighbourPointNormals, res.K, res.k, res.active);

      // gradient step
      const auto p_k = p + 0.5f * util::to_glm(grad.head(3));

      // project
      auto [p_final, n_final] = project_iterative(apss, p_k, gaussianStd);
      // update pointcloud / kdtree
      resampled.updatePoint(i, p_final, n_final);
    }
  }

  return resampled;
}
