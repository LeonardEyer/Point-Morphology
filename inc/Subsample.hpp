#pragma once

#include <ostream>
#include <random>
#include <ssg.hpp>

#include "Embedding.hpp"
#include "InverseIterative.hpp"
#include "PointCloud.hpp"
#include "Utils.hpp"

template <ImportanceEmbedding Embedding>
inline PointCloud subsample(const PointCloud &cloud, double radius,
                            const Embedding &embedding, float eps = 0.5) {

  auto sampled_positions = std::vector<PointCloud::Position>{};
  auto sampled_normals = std::vector<PointCloud::Normal>{};
  auto sampled = ssg::SSG<util::SSGAdaptor>(sampled_positions, 2*radius);

  // Avoid reallocating a bunch of memory
  auto mem = PreallocatedMemory(256);

  std::mt19937 rng(1337);
  std::vector<size_t> indices(cloud.positions.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);

  for (size_t i = 0; i < cloud.positions.size(); ++i) {

    size_t randIndex = indices[i];
    glm::vec3 p = cloud.positions[randIndex];
    glm::vec3 n = cloud.normals[randIndex];
    auto x = embedding(p, n);
    
    // we get sorted neighbours
    auto neighbors = sampled.inRadius(p, radius, true);

    double s = 1.0;
    if (!neighbors.empty()) {

      std::vector<Feature6D> neighborPointNormals;
      for (auto idx : neighbors) {
        neighborPointNormals.push_back(
            embedding(sampled_positions[idx], sampled_normals[idx]));
      }

      s = takeInverseIterative(x, neighborPointNormals, mem, rbfKernel, eps).s;
    }
    if (s > eps) {
      sampled.insert(p);
      sampled_normals.push_back(n);
    }
  }

  std::cout << "Subsampling Reduction = "
            << 1.0f - sampled_positions.size() /
                          static_cast<float>(cloud.positions.size())
            << " (" << cloud.positions.size() << " -> "
            << sampled_positions.size() << ")" << std::endl;

  return PointCloud(sampled_positions, sampled_normals);
}
