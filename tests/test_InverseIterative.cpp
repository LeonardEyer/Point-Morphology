#include "InverseIterative.hpp"

#include <gtest/gtest.h>

using V6f = Eigen::Vector<float, 6>;
using PointNormal = V6f;

struct Kernel {
  Eigen::MatrixXd K; // inverse kernel matrix
  Eigen::VectorXd k;
};

template <typename KernelFunc>
Kernel buildInverseKernelMatrix(const PointNormal &x,
                                const std::vector<PointNormal> &neighbours,
                                const KernelFunc &kernelFunc) {

  const auto N = neighbours.size();

  Eigen::MatrixXd K(N, N);
  Eigen::VectorXd k(N);

  for (auto i = 0; i < N; i++) {
    k(i) = kernelFunc(x, neighbours[i]);
  }

  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < N; j++) {
      K(i, j) = kernelFunc(neighbours[i], neighbours[j]);
    }
  }

  return {K.inverse(), k};
};

TEST(InverseIterativeTests, IterativeSameAsFull) {

  auto neighbours = std::vector<PointNormal>(10);
  for (auto &n : neighbours) {
    n = PointNormal::Random();
  }
  // Remark: do not use auto or we will get random values each time we read
  // point
  PointNormal point = PointNormal::Random();

  const auto kernelFunc = [](const auto &a, const auto &b) {
    return std::exp(-(a - b).squaredNorm());
  };

  auto inverseWhole = buildInverseKernelMatrix(point, neighbours, kernelFunc);
  auto inverseIterative = takeInverseIterative(point, neighbours, kernelFunc);

  auto expect = importance(inverseWhole.K, inverseWhole.k);
  auto actual = importance(inverseIterative.K, inverseIterative.k);

  EXPECT_EQ(expect, actual);
  EXPECT_NEAR(expect, inverseIterative.s, 1e-8);
}
