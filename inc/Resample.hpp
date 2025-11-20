#pragma once

#include "APSS.hpp"
#include "InverseIterative.hpp"
#include "PointCloud.hpp"

inline void resample(PointCloud &cloud, double gaussianStd, double sigmaP,
                     size_t iterations = 10) {

  auto apss = APSS(cloud);
  auto nPoints = cloud.positions.size();
  static const auto sigma = 1.0f;
  static double radius = gaussianStd * gaussianStd;

  for (auto iter = 0; iter < iterations; ++iter) {
    for (auto i = 0; i < nPoints; ++i) {
      std::cout << "Progress = "
                << ((iter + 1) * (i + 1)) /
                       static_cast<float>(iterations * nPoints)
                << std::endl;

      auto &p = cloud.positions[i];
      auto &n = cloud.normals[i];

      auto neighbours = cloud.neighboursInRadius(p, gaussianStd * sigmaP);

      if (neighbours.empty()) {
        std::cout << "No neighbours. skipping" << std::endl;
        continue;
      }

      std::vector<size_t> neighbourIndices;
      for (auto &[idx, _] : neighbours)
        neighbourIndices.push_back(idx);

      const auto [s, K, k] =
          takeInverseIterative(p, n, cloud, neighbourIndices, -1);

      // compute the (local) gradient of our weighting s (in 6D [p, n])
      const auto grad = [&] {
        auto gradient = Eigen::Vector<float, 6>{};
        auto x = Eigen::Vector<float, 6>{p.x, p.y, p.z, n.x, n.y, n.z};

        for (auto i = 0; i < neighbours.size(); ++i) {
          const auto &n_i_pos = cloud.positions[neighbours[i].first];
          const auto &n_i_norm = cloud.normals[neighbours[i].first];
          auto x_i =
              Eigen::Vector<float, 6>{n_i_pos.x,  n_i_pos.y,  n_i_pos.z,
                                      n_i_norm.x, n_i_norm.y, n_i_norm.z};

          for (auto j = 0; j < neighbours.size(); ++j) {

            const auto &n_j_pos = cloud.positions[neighbours[j].first];
            const auto &n_j_norm = cloud.normals[neighbours[j].first];

            auto x_j =
                Eigen::Vector<float, 6>{n_j_pos.x,  n_j_pos.y,  n_j_pos.z,
                                        n_j_norm.x, n_j_norm.y, n_j_norm.z};

            gradient += (2 * x - x_i - x_j) * k(i) * k(j) * K(i, j);
          }
        }

        return (2 / sigma) * gradient;
      }();

      // gradient step
      auto p_k = p + 0.5f * PointCloud::Position(grad.x(), grad.y(), grad.z());

      // project
      auto fitted = apss.fit(p_k, radius);
      p_k =
          std::visit([&p](const auto &fit) { return project(fit, p); }, fitted);
      auto n_k = std::visit([&p](const auto &fit) { return gradient(fit, p); },
                            fitted);

      // update pointcloud / kdtree
      cloud.deletePoint(i);
      cloud.insertPoint(p_k, n_k);
    }
  }
}
