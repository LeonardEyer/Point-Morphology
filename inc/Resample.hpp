#pragma once

#include "APSS.hpp"
#include "InverseIterative.hpp"
#include "PointCloud.hpp"

inline PointCloud resample(const PointCloud &cloud, double gaussianStd,
                           float sigmaP, float sigmaN, size_t iterations = 10) {

  // apply sigmaN sigmaP
  PointCloud resampled = [sigmaP, sigmaN, &cloud] {
    auto positions = cloud.positions;
    std::transform(positions.begin(), positions.end(), positions.begin(),
                   [sigmaP](auto x) { return x / sigmaP; });

    auto normals = cloud.normals;
    std::transform(normals.begin(), normals.end(), normals.begin(),
                   [sigmaN](auto x) { return x / sigmaN; });

    return PointCloud(positions, normals);
  }();

  constexpr auto sigma = 1.0f;

  using V6f = Eigen::Vector<float, 6>;
  auto apss = APSS(resampled);
  auto nPoints = resampled.positions.size();

  for (auto iter = 0; iter < iterations; ++iter) {
    for (auto i = 0; i < nPoints; ++i) {

      if (i != 0 && i % 1000 == 0) {
        std::cout << "Progress = "
                  << ((iter + 1) * (i + 1)) /
                         static_cast<float>(iterations * nPoints)
                  << std::endl;
      }

      auto &p = resampled.positions[i];
      auto &n = resampled.normals[i];

      auto neighbours =
          resampled.tree->neighboursInRadius(p, gaussianStd * sigmaP);

      if (neighbours.empty()) {
        // std::cout << "No neighbours. skipping" << std::endl;
        continue;
      }

      std::cout << "neighbours = [";
      for (const auto [_, dist] : neighbours) {
        std::cout << dist << ", ";
      }
      std::cout << "]" << std::endl;

      std::vector<size_t> neighbourIndices;
      for (auto &[idx, _] : neighbours)
        neighbourIndices.push_back(idx);

      const auto res = takeInverseIterative(p, n, cloud, neighbourIndices, -1);

      const auto &[s, K, k, _] = res;

      std::cout << "k = [";
      for (auto i = 0; i < neighbours.size(); i++) {
        std::cout << k(i) << ", ";
      }
      std::cout << "]" << std::endl;

      // compute the (local) gradient of our weighting s (in 6D [p, n])
      const V6f grad = [&]() -> V6f {
        V6f gradient = V6f::Zero();
        V6f x = concat(p, n);

        for (auto i = 0; i < neighbours.size(); ++i) {
          auto neighbour_i = neighbours[i].first;
          V6f x_i = concat(resampled.positions[neighbour_i],
                           resampled.normals[neighbour_i]);

          for (auto j = 0; j < neighbours.size(); ++j) {
            auto neighbour_j = neighbours[j].first;
            V6f x_j = concat(resampled.positions[neighbour_j],
                             resampled.normals[neighbour_j]);

            gradient += (2 * x - x_i - x_j) * k(i) * k(j) * K(i, j);
          }
        }
        gradient *= (-2.0 / sigma * sigma);

        return gradient;
      }();

      if (i != 0 && i % 1000 == 0) {
        std::cout << "grad = (" << grad.x() << ", " << grad.y() << ", "
                  << grad.z() << ")" << std::endl;
      }

      // gradient step
      const auto p_k =
          p + 0.5f * PointCloud::Position(grad.x(), grad.y(), grad.z());

      // project
      auto [p_final, n_final] = project_iterative(apss, p_k, gaussianStd);
      // update pointcloud / kdtree
      resampled.updatePoint(i, p_final, n_final);
    }
  }

  // undo sigmaN sigmaP scaling
  {
    std::transform(resampled.positions.begin(), resampled.positions.end(),
                   resampled.positions.begin(),
                   [sigmaP](auto x) { return x * sigmaP; });

    std::transform(resampled.normals.begin(), resampled.normals.end(),
                   resampled.normals.begin(),
                   [sigmaN](auto x) { return x * sigmaN; });
  };

  return resampled;
}
