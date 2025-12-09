#pragma once

#include <random>

#include "APSS.hpp"
#include "InverseIterative.hpp"
#include "PointCloud.hpp"
#include "Utils.hpp"

inline PointCloud resample(const PointCloud &cloud, double gaussianStd,
                           float sigmaP, float sigmaN, size_t iterations = 10) {

  constexpr auto sigma = 1.0f;

  auto resampled_positions = cloud.positions;
  auto resampled_normals = cloud.normals;

  auto resampled = PointKDTree<false>(resampled_positions);

  const auto apss = APSS(cloud);
  const auto nPoints = resampled_positions.size();

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, nPoints - 1);

  auto mem = PreallocatedMemory(256);

  for (auto iter = 0; iter < iterations; ++iter) {

    // Create a list of indices 0 .. nPoints-1
    std::vector<size_t> indices(nPoints);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle them so each index is used exactly once in random order
    std::shuffle(indices.begin(), indices.end(), rng);

    for (auto i = 0; i < nPoints; ++i) {

      if (i != 0 && i % 10000 == 0) {
        std::cout << "Progress = "
                  << (iter * nPoints + i + 1) /
                         static_cast<float>(iterations * nPoints)
                  << std::endl;
      }
      // choose a random index
      size_t randIndex = indices[i];

      const auto &p = resampled_positions[randIndex];
      const auto &n = resampled_normals[randIndex];
      const auto x = util::concat(p / sigmaP, n / sigmaN);

      auto neighbours = resampled.neighboursInRadius(p, gaussianStd);

      if (neighbours.empty()) {
        // std::cout << "No neighbours. skipping" << std::endl;
        continue;
      }

      std::vector<util::PointNormal> neighbourPointNormals;
      for (auto &[idx, _] : neighbours) {
        neighbourPointNormals.push_back(
            util::concat(resampled_positions[idx] / sigmaP,
                         resampled_normals[idx] / sigmaN));
      }

      const auto res =
          takeInverseIterative(x, neighbourPointNormals, mem, rbfKernel, -1);

      const auto grad =
          importance_grad(x, neighbourPointNormals, res.K, res.k, res.active);

      // gradient step
      const auto p_k = p - 0.5f * util::to_glm(grad.head(3));

      // project
      auto [p_final, n_final] = project_iterative(apss, p_k, gaussianStd);

      // update pointcloud / kdtree
      {
        resampled_positions[randIndex] = p_final;
        resampled_normals[randIndex] = n_final;

        // resampled.adaptor.index->removePoint(randIndex);
        // resampled.addPoints(randIndex, randIndex);
      }
    }
  }

  return PointCloud(resampled_positions, resampled_normals);
}
