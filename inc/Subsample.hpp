#pragma once

#include "InverseIterative.hpp"
#include "PointCloud.hpp"
#include "Embedding.hpp"

template <ImportanceEmbedding Embedding>
inline PointCloud subsample(const PointCloud &cloud, double distance,
                            const Embedding &embedding, float eps = 0.5) {

  
  auto sampled_positions = std::vector<PointCloud::Position>{};
  auto sampled_normals = std::vector<PointCloud::Normal>{};
  PointKDTree sampled = PointKDTree<true>(sampled_positions);

  // Avoid reallocating a bunch of memory
  auto mem = PreallocatedMemory(256);

  for (size_t i = 0; i < cloud.positions.size(); ++i) {

    glm::vec3 p = cloud.positions[i];

    // we get sorted neighbours
    auto neighbors = sampled.neighboursInRadius(p, distance);

    double s = 1.0;

    glm::vec3 n = cloud.normals[i];
    auto x = embedding(p, n);
    if (!neighbors.empty()) {
      
      std::vector<Feature6D> neighborPointNormals;
      for (auto &[idx, _] : neighbors) {
        neighborPointNormals.push_back(embedding(sampled_positions[idx], sampled_normals[idx]));
      }

      s = takeInverseIterative(x, neighborPointNormals, mem, rbfKernel, eps).s;
      
    }
    if (s > eps) {
      sampled_positions.push_back(p);
      sampled_normals.push_back(n);
      const auto newIndex = sampled_positions.size() - 1;
      sampled.addPoints(newIndex, newIndex);
    }
  }

  std::cout << "Reduction = "
            << 1.0f - sampled_positions.size() /
                          static_cast<float>(cloud.positions.size())
            << std::endl;

  return PointCloud(sampled_positions, sampled_normals);
}
