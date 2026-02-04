#pragma once

#include <glm/geometric.hpp>
#include <random>
#include <ssg.hpp>

#include "APSS.hpp"
#include "Embedding.hpp"
#include "InverseIterative.hpp"
#include "PointCloud.hpp"
#include "Utils.hpp"

struct SSGAdaptor {
  using VectorType = glm::vec3;
  using DataType = std::vector<VectorType>;
  using IndexType = size_t;

  static float get_x(const VectorType &v) { return v[0]; }
  static float get_y(const VectorType &v) { return v[1]; }
  static float get_z(const VectorType &v) { return v[2]; }
  static const VectorType &get_vec(const DataType &data, IndexType idx) {
    return data[idx];
  }
  static float distance2(const VectorType &a, const VectorType &b) {
    float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
  }

  static IndexType insert(DataType &d, const VectorType &v) {
    d.push_back(v);
    return d.size() - 1;
  }

  static void update(DataType &d, IndexType idx, const VectorType &v) {
    d[idx] = v;
  }
};

template <ImportanceEmbedding Embedding, typename ProjectionFn>
inline PointCloud resample(const PointCloud &cloud, double gaussianStd,
                           const Embedding &embedding,
                           const ProjectionFn &project,
                           size_t iterations = 10) {

  constexpr auto sigma = 1.0f;

  auto resampled_positions = cloud.positions;
  auto resampled_normals = cloud.normals;

  std::cout << "Resampling n = " << cloud.positions.size() << " positions"
            << std::endl;

  // auto resampled2 = PointKDTree<false>(resampled_positions);

  auto resampled = ssg::SSG<SSGAdaptor>(resampled_positions, gaussianStd);

  const auto nPoints = resampled_positions.size();

  std::mt19937 rng(1337);

  auto mem = PreallocatedMemory(256);

  for (auto iter = 0; iter < iterations; ++iter) {

    // Create a list of indices 0 .. nPoints-1
    std::vector<size_t> indices(nPoints);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle them so each index is used exactly once in random order
    std::shuffle(indices.begin(), indices.end(), rng);

    for (auto i = 0; i < nPoints; ++i) {
      size_t randIndex = indices[i];
      const auto &p = resampled_positions[randIndex];

      if (i != 0 && i % 10000 == 0) {
      std::cout << "Progress (i = " << i << ", iter = " << iter << ") = "
                << (iter * nPoints + i + 1) /
                       static_cast<float>(iterations * nPoints)
                << ", index = " << randIndex << " p = [" << p.x << ", " << p.y
                << ", " << p.z << "]" << std::endl;
      }
      // choose a random index
      auto neighbours = resampled.inRadius(p, gaussianStd, true);

      if (neighbours.empty()) {
        continue;
      }

      const auto &n = resampled_normals[randIndex];
      const auto x = embedding(p, n);

      std::vector<util::Feature6D> neighbourFeatures;
      for (auto idx : neighbours) {
        neighbourFeatures.push_back(
            embedding(resampled_positions[idx], resampled_normals[idx]));
      }

      const auto [_, K, k, active] =
          takeInverseIterative(x, neighbourFeatures, mem, rbfKernel, -1);

      const auto grad = importance_grad(x, neighbourFeatures, K, k, active);

      // std::cout << "Done grad" << std::endl;
      auto grad_p = util::to_glm(grad.head(3));

      if (glm::length2(grad_p) >= 1) {
	grad_p = glm::normalize(grad_p) * 0.1f;
      }

      // gradient step
      const auto p_k = p - 0.5f * grad_p;

      // project
      auto [p_final, n_final] = project(p_k);

      // update pointcloud
      resampled.update(randIndex, p_final);
      resampled_normals[randIndex] = n_final;
    }
  }

  return PointCloud(resampled_positions, resampled_normals);
}
