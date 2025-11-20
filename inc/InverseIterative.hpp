#pragma once

#include "PointCloud.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <glm/glm.hpp>
#include <vector>

inline double sqDist(const glm::vec3 &a, const glm::vec3 &b) noexcept {
  glm::dvec3 diff = glm::dvec3(a) - glm::dvec3(b);
  return glm::dot(diff, diff);
}

inline double kernel(double dist2) noexcept { return std::exp(-dist2); }

inline auto concat(const glm::vec3 &a, const glm::vec3 &b) {
  return Eigen::Vector<float, 6>{a.x, a.y, a.z, b.x, b.y, b.z};
}

struct KernelInverseResult {
  double s;          // weight
  Eigen::MatrixXd K; // inverted kernel matrix K^{-1}
  Eigen::VectorXd k; // kernel values from point to each neighbour
};

inline KernelInverseResult takeInverseIterative(
    const glm::vec3 &xPos, const glm::vec3 &xNorm, const PointCloud &cloud,
    const std::vector<size_t> &neighborIndices, double threshold) {

  // we want to compute s(x) = 1 − k^T K^{-1} k / k(x,x)
  // by iteratively inverting K
  // x := [p / sigma_p, n / sigma_n] // point plus normal normalized

  const int num = static_cast<int>(neighborIndices.size());
  if (num <= 0)
    return {0.0};

  Eigen::MatrixXd Kn = Eigen::MatrixXd::Zero(num, num);
  Eigen::VectorXd kx = Eigen::VectorXd::Zero(num);
  Eigen::VectorXd kn_np1(num);
  Eigen::VectorXd e(num);

  constexpr double dpTH = 1e-5;
  double p = 0.0;

  auto sqDistVM = [&](int idx) {
    auto x = concat(xPos, xNorm);
    auto xi = concat(cloud.positions[idx], cloud.normals[idx]);
    return (x - xi).squaredNorm();
  };

  auto sqDistMM = [&](int i, int j) {
    auto xi = concat(cloud.positions[i], cloud.normals[i]);
    auto xj = concat(cloud.positions[j], cloud.normals[j]);
    return (xi - xj).squaredNorm();
  };

  // Initialize first point
  Kn(0, 0) = 1.0;
  kx(0) = kernel(sqDistVM(neighborIndices[0]));
  p = Kn(0, 0) * kx(0) * kx(0);

  unsigned int k = 0;
  int index = 0;

  while (index < num - 1 && (1.0 - p) >= threshold) {
    ++k;
    ++index;

    // compute kn_np1
    for (unsigned int m = 0; m < k; ++m)
      kn_np1(m) = kernel(sqDistMM(neighborIndices[index], neighborIndices[m]));

    // e = Kn * kn_np1 (Eigen handles matrix-vector multiplication)
    e.head(k) = Kn.topLeftCorner(k, k) * kn_np1.head(k);

    double dot = kn_np1.head(k).dot(e.head(k));
    if ((1.0 - dot) < dpTH) {
      --k;
      continue;
    }

    double dp = 1.0 / (1.0 - dot);

    // Update Kn matrix
    Kn.topLeftCorner(k, k) += dp * e.head(k) * e.head(k).transpose();
    Kn.topRightCorner(k, 1) = -dp * e.head(k);
    Kn.bottomLeftCorner(1, k) = Kn.topRightCorner(k, 1).transpose();
    Kn(k, k) = dp;

    double f = kx.head(k).dot(e.head(k));
    kx(k) = kernel(sqDistVM(neighborIndices[index]));

    double error = dp * (f - kx(k)) * (f - kx(k));
    p += error;
  }

  return {p, Kn, kx};
}
