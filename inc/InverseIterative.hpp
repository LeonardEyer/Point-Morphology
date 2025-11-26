#pragma once

#include "PointCloud.hpp"
#include "polyscope/view.h"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <cmath>
#include <glm/glm.hpp>
#include <vector>

inline double sqDist(const glm::vec3 &a, const glm::vec3 &b) noexcept {
  return glm::distance2(a, b);
}

inline double kernel(double dist2) noexcept { return std::exp(-dist2); }

inline auto concat(const glm::vec3 &a, const glm::vec3 &b) {
  return Eigen::Vector<float, 6>{a.x, a.y, a.z, b.x, b.y, b.z};
}

struct KernelInverseResult {
  double s;          // weight
  Eigen::MatrixXd K; // inverted kernel matrix K^{-1}
  Eigen::VectorXd k; // kernel values from point to each neighbour
  Eigen::VectorX<bool> active;
};

KernelInverseResult takeInverseIterative(
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
  Eigen::VectorXd an(num);
  Eigen::VectorX<bool> active = Eigen::VectorX<bool>::Ones(num);

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

    // compute k_n(x_{n+1}) = (k_n(x_{n+1}))_i = k(x_{n+1}, x_i) for i ≤ n
    for (unsigned int m = 0; m < k; ++m)
      kn_np1(m) = kernel(sqDistMM(neighborIndices[index], neighborIndices[m]));

    // a_n = K_n * k_n(x_{n+1})
    an.head(k) = Kn.topLeftCorner(k, k) * kn_np1.head(k);

    double dot = kn_np1.head(k).dot(an.head(k));
    if ((1.0 - dot) < dpTH) {
      --k;
      continue;
    }

    // g_n = (k(x_{n+1},x_{n+1}) - k_n(x_{n+1})^T K_n^{-1} k_n(x_{n+1}))^{−1}
    double gn = 1.0 / (1.0 - dot);

    // Update Kn matrix
    Kn.topLeftCorner(k, k) += gn * an.head(k) * an.head(k).transpose();
    Kn.col(k).head(k) = -gn * an.head(k);
    Kn.row(k).head(k) = -gn * an.head(k);

    Kn(k, k) = gn;

    if (threshold == -1) {
      std::cout << "k = " << k << std::endl;
      std::cout << "x = [" << xPos.x << ", " << xPos.y << ", " << xPos.z << "]"
                << std::endl;
      std::cout << "n = [" << xNorm.x << ", " << xNorm.y << ", " << xNorm.z
                << "]" << std::endl;
      std::cout << "x_j = [" << cloud.positions[index].x << ", "
                << cloud.normals[index].y << ", " << cloud.normals[index].z
                << "]" << std::endl;
      std::cout << "n_j = [" << cloud.normals[index].x << ", "
                << cloud.normals[index].y << ", " << cloud.normals[index].z
                << "]" << std::endl;
      std::cout << "j = " << index << std::endl;
      std::cout << "Distance = " << sqDistVM(neighborIndices[index])
                << std::endl;
      std::cout << "|x - x_j|^2 = "
                << glm::distance2(xPos, cloud.positions[index]) << std::endl;
      std::cout << "|n _ n_j|^2 = "
                << glm::distance2(xNorm, cloud.normals[index]) << std::endl;
      std::cout << "k([x, n], [x_j, n_j]) = "
                << kernel(sqDistVM(neighborIndices[index])) << std::endl;
    }

    kx(k) = kernel(sqDistVM(neighborIndices[index]));

    double f = kx.head(k).dot(an.head(k));
    double error = gn * (f - kx(k)) * (f - kx(k));
    p += error;
  }

  return {p, Kn, kx, active};
}

using V6f = Eigen::Vector<float, 6>;
using PointNormal = V6f;
using PointNormals = std::vector<PointNormal>;

float kernel6D(V6f x, V6f y) { return std::exp(-(x - y).squaredNorm()); }

template <typename KernelFunc = decltype(&kernel6D)>
KernelInverseResult
takeInverseIterative(const PointNormal &x, const PointNormals &neighbours,
                     const KernelFunc &kernelFunc = kernel6D,
                     double threshold = -1) {

  // we want to compute s(x) = 1 − k^T K^{-1} k / k(x,x)
  // by iteratively inverting K
  // x := [p / sigma_p, n / sigma_n] // point plus normal normalized

  const int N = static_cast<int>(neighbours.size());
  if (N <= 0)
    return {0.0};

  Eigen::MatrixXd Kn = Eigen::MatrixXd::Zero(N, N);
  Eigen::VectorXd kx = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd kn_np1(N);
  Eigen::VectorXd an(N);
  Eigen::VectorX<bool> active_mask = Eigen::VectorX<bool>::Ones(N);

  constexpr double dpTH = 1e-5;
  double p = 0.0;

  // Initialize first point
  Kn(0, 0) = 1.0;
  kx(0) = kernelFunc(x, neighbours[0]);
  p = Kn(0, 0) * kx(0) * kx(0);

  unsigned int k = 0;
  int index = 0;

  while (index < N - 1 && (1.0 - p) >= threshold) {
    ++k;
    ++index;

    // compute k_n(x_{n+1}) = (k_n(x_{n+1}))_i = k(x_{n+1}, x_i) for i ≤ n
    for (unsigned int m = 0; m < k; ++m)
      kn_np1(m) = kernelFunc(neighbours[index], neighbours[m]);

    // a_n = K_n * k_n(x_{n+1})
    an.head(k) = Kn.topLeftCorner(k, k) * kn_np1.head(k);

    double dot = kn_np1.head(k).dot(an.head(k));
    if ((1.0 - dot) < dpTH) {
      // mark this neighbour as inactive
      active_mask[index] = false;
      --k;
      continue;
    }

    // g_n = (k(x_{n+1},x_{n+1}) - k_n(x_{n+1})^T K_n^{-1} k_n(x_{n+1}))^{−1}
    double gn = 1.0 / (1.0 - dot);

    if (false && threshold == -1) {
      std::cout << "Kn(" << k << ", " << k << ") = [\n"
                << Kn.topLeftCorner(k, k) << "]" << std::endl;
    }

    // Update Kn matrix
    Kn.topLeftCorner(k, k) += gn * an.head(k) * an.head(k).transpose();
    Kn.col(k).head(k) = -gn * an.head(k);
    Kn.row(k).head(k) = -gn * an.head(k);

    Kn(k, k) = gn;

    kx(k) = kernelFunc(x, neighbours[index]);

    if (false && threshold == -1) {
      std::cout << "Kn(" << k << ", " << k << ") = [\n"
                << Kn.topLeftCorner(k + 1, k + 1) << "]" << std::endl;
    }

    if (false && threshold == -1) {
      std::cout << "j = " << index << std::endl;
      std::cout << "k = " << k << std::endl;
      std::cout << "k(x, x_j) = " << kernelFunc(x, neighbours[index])
                << std::endl;
    }

    double f = kx.head(k).dot(an.head(k));
    double error = gn * (f - kx(k)) * (f - kx(k));
    p += error;
  }

  return {p, Kn.topLeftCorner(k + 1, k + 1), kx.head(k + 1), active_mask};
}
