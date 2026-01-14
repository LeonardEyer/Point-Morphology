#pragma once

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <cassert>
#include <cmath>
#include <csignal>
#include <iostream>
#include <vector>


template <typename T> inline float rbfKernel(const T &x, const T &y) {
  return std::exp(-(x - y).squaredNorm());
}

struct KernelInverseResult {
  using Mask = Eigen::VectorX<bool>;

  double s;          // Iterartive approximation to the importance of this point
  Eigen::MatrixXd K; // Inverted kernel matrix
  Eigen::VectorXd k; // k(x, x_i) point similarity with each neighbour
  Mask active; // collection of neighbours considered (we skip similar ones)
};

inline float importance(const Eigen::MatrixXd &K, const Eigen::VectorXd &k) {
  // Actually we need to divide by k(x,x) for non normalized kernels
  return 1.0 - k.transpose() * K * k;
};

template <typename T>
T importance_grad(const T &x, const std::vector<T> &neighbours,
                  const Eigen::MatrixXd &K, const Eigen::VectorXd &k,
                  const Eigen::VectorX<bool> &active, float sigma = 1.0f) {

  // The K matrix only contais the k first active neighbours. Therefore we need
  // a way of indexing the active ones
  auto neighboursFiltered = std::vector<T>();
  for (auto i = 0; i < neighbours.size(); i++) {
    if (!active[i]) {
      continue; // skip inactive
    }
    neighboursFiltered.push_back(neighbours[i]);
  }

  T sgrad = T::Zero();
  for (auto i = 0; i < neighboursFiltered.size(); i++) {
    for (auto j = 0; j < neighboursFiltered.size(); j++) {

      const auto &x_i = neighboursFiltered[i];
      const auto &x_j = neighboursFiltered[j];

      sgrad += (2 * x - x_i - x_j) * k(i) * k(j) * K(i, j);
    }
  }
  return (-2.0f / (sigma * sigma)) * sgrad;
}

// This is a performance improvement measure. Since the function
// `takeInverseIterative` will be called many times in a loop with potentially
// many neighbbours (N) we do not want to allocate/deallocate so many times
struct PreallocatedMemory {
  Eigen::MatrixXd K;
  Eigen::VectorXd kx;
  Eigen::VectorXd kn_np1;
  Eigen::VectorXd an;

  PreallocatedMemory(size_t N) : K(N, N), kx(N), kn_np1(N), an(N) {};

  void grow(float factor = 1.2) {
    // Compute new size + %
    auto newSize = static_cast<Eigen::Index>(std::ceil(factor * kx.size()));

    // Resize matrices/vectors conservatively to keep existing data
    K.conservativeResize(newSize, newSize);
    kx.conservativeResize(newSize);
    kn_np1.conservativeResize(newSize);
    an.conservativeResize(newSize);

    // Optionally initialize new entries
    K.bottomRows(newSize - K.rows()).setZero();
    K.rightCols(newSize - K.cols()).setZero();
    kx.tail(newSize - kx.size()).setZero();
    kn_np1.tail(newSize - kn_np1.size()).setZero();
    an.tail(newSize - an.size()).setZero();
  }
};

template <typename T, typename KernelFunc = decltype(&rbfKernel<T>)>
KernelInverseResult takeInverseIterative(
    const T &x, const std::vector<T> &neighbours, PreallocatedMemory &prealloc,
    const KernelFunc &kernelFunc = rbfKernel, double threshold = -1) {

  // we want to compute s(x) = 1 − k^T K^{-1} k / k(x,x)
  // by iteratively inverting K

  const int N = static_cast<int>(neighbours.size());
  if (N <= 0)
    return {1.0};

  auto &[Kn, kx, kn_np1, an] = prealloc;

  // Since we obtain vectors possibly from last iteration Zero out accumulation
  // variables.
  Kn.setZero();
  kx.setZero();

  Eigen::VectorX<bool> active_mask = Eigen::VectorX<bool>::Zero(N);

  // we assume a normalized kernel
  assert(kernelFunc(x, x) == 1);

  constexpr double dpTH = 1e-5;
  double p = 0.0;

  // Initialize first point
  Kn(0, 0) = 1.0;
  kx(0) = kernelFunc(x, neighbours[0]);
  active_mask[0] = true;

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
      --k;
      continue;
    }

    // Grow as soon as we run out of memory
    if (k >= kx.size()) {
      std::cout << "More neighbours than expected. (kx.size() = " << kx.size()
                << ", k = " << k << ") Growing matrices" << std::endl;
      prealloc.grow(1.2);
    }

    // Calculate scalar g_n based on the Schur complement definition in the
    // paper g_n = (k(x_{n+1},x_{n+1}) - k_n(x_{n+1})^T K_n^{-1}
    // k_n(x_{n+1}))^{−1}
    double gn = 1.0 / (1.0 - dot);

    // Iterative Block Matrix Inversion (Eq. 10)
    // We are constructing K_{n+1}^-1 by updating K_n^-1 (Knk) and padding the
    // new row/col. The target structure is:
    // [ K_n^-1 + g_n * a_n * a_n^T    -g_n * a_n ]
    // [      -g_n * a_n^T                 g_n    ]
    const auto an_head = an.head(k);

    // 1. Update Top-Left Block: K_n^{-1} + g_n * a_n * a_n^T
    Kn.topLeftCorner(k, k).noalias() += gn * an_head * an_head.transpose();

    // 2. Update Top-Right Block: -g_n * a_n
    Kn.col(k).head(k).noalias() = -gn * an_head;

    // 3. Update Bottom-Left Block: -g_n * a_n^T
    Kn.row(k).head(k).noalias() = (-gn * an_head).transpose();

    // 4. Update Bottom-Right Element: g_n
    Kn(k, k) = gn;

    kx(k) = kernelFunc(x, neighbours[index]);
    active_mask[index] = true; // mark this neighbour as active
    double f = kx.head(k).dot(an.head(k));
    double error = gn * (f - kx(k)) * (f - kx(k));
    p += error;
  }

  // Here we copy a slice of the kernel matrix. Potentially we could speed up
  // iterations by returning a reference? / not copying at all?
  return {1.0 - p, Kn.topLeftCorner(k + 1, k + 1), kx.head(k + 1), active_mask};
}

// A method for when we do not want to prallocate. Simply allocate the maximum
// amount of memory required given the number of neighbours
template <typename T, typename KernelFunc = decltype(&rbfKernel<T>)>
KernelInverseResult
takeInverseIterative(const T &x, const std::vector<T> &neighbours,
                     const KernelFunc &kernelFunc = rbfKernel,
                     double threshold = -1) {
  const int N = static_cast<int>(neighbours.size());

  auto mem = PreallocatedMemory(N);
  return takeInverseIterative(x, neighbours, mem, kernelFunc, threshold);
}
