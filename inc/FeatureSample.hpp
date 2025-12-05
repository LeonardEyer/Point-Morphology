#pragma once

#include "PointCloud.hpp"
#include "Utils.hpp"
#include <cassert>
#include <csignal>
#include <optional>

inline std::optional<std::pair<PointCloud::Position, PointCloud::Normal>>
projectToFeature(const PointCloud::Position &p, const PointCloud::Normal &n,
                 const Eigen::Matrix4f &A_qem,
                 float featureRatioThreshold = 1000.0f) {
  // 1. Extract the 3x3 matrix A and the 3x1 vector b from the 4x4 QEM matrix
  // A_qem = [ A   b ]
  //         [ b^T c ]
  // The minimization problem is A * x = -b
  Eigen::Matrix3f A = A_qem.block<3, 3>(0, 0);
  Eigen::Vector3f b = A_qem.block<3, 1>(0, 3);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(A);
  if (solver.info() != Eigen::Success) {
    return std::nullopt;
  }

  // Sorted in increasing order
  Eigen::Vector3f eigenvalues = solver.eigenvalues();
  Eigen::Matrix3f eigenvectors = solver.eigenvectors();

  if (eigenvalues[1] / eigenvalues[0] > featureRatioThreshold) {

    Eigen::Vector3f d_qem = eigenvectors.col(0);
    Eigen::Vector3f p_qem_eigen = Eigen::Vector3f::Zero();

    for (int i = 1; i < 3; ++i) { // i=1,2 (skipping 0, the feature direction)
      float lambda = eigenvalues[i];
      Eigen::Vector3f v = eigenvectors.col(i);
      if (lambda > 1e-6f) {
        p_qem_eigen += (v.dot(-b) / lambda) * v;
      }
    }

    // project onto line [p_qem, d_qem]
    const auto [p_projected, n_projected] = [p_qem = util::to_glm(p_qem_eigen),
                                             d_qem = util::to_glm(d_qem), &p,
                                             &n, &eigenvectors] {
      // Projection: P' = P_line + dot(P_orig - P_line, dir) * dir
      glm::vec3 p_projected = p_qem + glm::dot(p - p_qem, d_qem) * d_qem;

      // ensure we have nonzero length
      assert(glm::dot(n, d_qem) * d_qem > 1e-8f);
      glm::vec3 n_projected = glm::normalize(n - glm::dot(n, d_qem) * d_qem);

      return std::pair{p_projected, n_projected};
    }();

    return std::pair<PointCloud::Position, PointCloud::Normal>{p_projected,
                                                               n_projected};
  }

  return std::nullopt;
}

inline auto edge_sample(const PointCloud &cloud) {
  std::vector<PointCloud::Position> edge_sampling_points;
  std::vector<PointCloud::Normal> edge_sampling_normals;
  for (auto i = 0; i < cloud.positions.size(); i++) {
    if (curvature_estimate(cloud, i) < 0.75) {
      const auto &p_i = cloud.positions[i];
      const auto &n_i = cloud.normals[i];
      const auto A_qem = quadric(cloud, i);
      const auto projected = projectToFeature(p_i, n_i, A_qem);

      if (projected) {
        const auto [projected_p, projected_n] = *projected;
        edge_sampling_points.push_back(projected_p);
        edge_sampling_normals.push_back(projected_n);
      }
    }
  }
  return PointCloud(edge_sampling_points, edge_sampling_normals);
}
