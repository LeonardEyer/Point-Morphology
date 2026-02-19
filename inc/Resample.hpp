#pragma once

#include <glm/geometric.hpp>
#include <random>
#include <ssg.hpp>

#include "APSS.hpp"
#include "Embedding.hpp"
#include "InverseIterative.hpp"
#include "PointCloud.hpp"
#include "Utils.hpp"

template <ImportanceEmbedding Embedding, typename ProjectionFn>
inline PointCloud
resample(const PointCloud &cloud, double radius, const Embedding &embedding,
         const ProjectionFn &project, size_t iterations = 10) {

  constexpr auto sigma = 1.0f;

  auto resampled_positions = cloud.positions;
  auto resampled_normals = cloud.normals;

  std::cout << "Resampling n = " << cloud.positions.size() << " positions"
            << std::endl;

  auto resampled = ssg::SSG<util::SSGAdaptor>(resampled_positions, radius);

  const auto nPoints = resampled_positions.size();

  std::vector<util::Feature6D> embeddings(nPoints);
  for (auto i = 0; i < nPoints; ++i) {
    embeddings[i] = embedding(resampled_positions[i], resampled_normals[i]);
  };

  std::mt19937 rng(1337);

  auto mem = PreallocatedMemory(256);
  for (auto iter = 0; iter < iterations; ++iter) {

    // Create a list of indices 0 .. nPoints-1
    std::vector<size_t> indices(nPoints);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle them so each index is used exactly once in random order
    std::shuffle(indices.begin(), indices.end(), rng);

    for (auto i = 0; i < nPoints; ++i) {

      if (i != 0 && i % 10000 == 0) {
        std::cout << "Progress (i = " << i << ", iter = " << iter << ") = "
                  << (iter * nPoints + i + 1) * 100 /
                         static_cast<float>(iterations * nPoints)
                  << "%" << std::endl;
      }
      size_t randIndex = indices[i];
      const auto &p = resampled_positions[randIndex];

      // choose a random index
      auto neighbours = resampled.inRadius(p, radius, true);

      if (neighbours.empty()) {
        continue;
      }

      const auto x = embeddings[randIndex];

      std::vector<util::Feature6D> neighbourFeatures;
      for (auto idx : neighbours) {
        neighbourFeatures.push_back(embeddings[idx]);
      }

      const auto [_, K, k, active] =
          takeInverseIterative(x, neighbourFeatures, mem, rbfKernel, -1);

      const auto grad = importance_grad(x, neighbourFeatures, K, k, active);

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

      embeddings[randIndex] = embedding(p_final, n_final);
    }
  }

  return {resampled_positions, resampled_normals};
}
