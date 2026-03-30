#pragma once

#include "APSS.hpp"
#include "Draw.hpp"
#include "PointCloud.hpp"
#include "PointStructuringElement.hpp"
#include <optional>

namespace morphology {

namespace detail {

template <typename Func>
void volume_apply(const Func &func, const ::detail::Bounds &bounds,
                  float spacing) {

  unsigned numThreads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  std::atomic<size_t> voxelsDone = 0;

  glm::vec3 min = bounds.first;
  glm::vec3 max = bounds.second;
  glm::vec3 size = max - min;
  int pointsX = std::max(1, int(std::floor(size.x / spacing)) + 1);
  int pointsY = std::max(1, int(std::floor(size.y / spacing)) + 1);
  int pointsZ = std::max(1, int(std::floor(size.z / spacing)) + 1);

  glm::vec3 step = glm::vec3(spacing);
  auto worker = [&](size_t zStart, size_t zEnd) {
    for (int z = zStart; z < zEnd; ++z) {
      for (int y = 0; y < pointsY; ++y) {
        for (int x = 0; x < pointsX; ++x) {
          glm::vec3 point = min + glm::vec3(x, y, z) * step;
          size_t idx = z * pointsY * pointsX + y * pointsX + x;
          func(idx, point);
          voxelsDone.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  };

  size_t chunk = pointsZ / numThreads;
  size_t zStart = 0;
  for (unsigned t = 0; t < numThreads; ++t) {
    size_t zEnd = (t == numThreads - 1) ? pointsZ : zStart + chunk;
    threads.emplace_back(worker, zStart, zEnd);
    zStart = zEnd;
  }

  size_t totalVoxels = pointsX * pointsY * pointsZ;
  while (voxelsDone < totalVoxels) {
    std::cout << "\rvolume_apply (Progress): "
              << (100.0 * voxelsDone / totalVoxels) << "%   " << std::flush;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  for (auto &t : threads)
    t.join();
  std::cout << "\rvolume_apply (Progress): 100%   \n";
}

template <typename Filter>
std::vector<glm::vec3> sampleBoxVolume(const ::detail::Bounds &bounds,
                                       float spacing, const Filter &filter) {

  // Remove duplicate code-----
  glm::vec3 min = bounds.first;
  glm::vec3 max = bounds.second;
  glm::vec3 size = max - min;
  int pointsX = std::max(1, int(std::floor(size.x / spacing)) + 1);
  int pointsY = std::max(1, int(std::floor(size.y / spacing)) + 1);
  int pointsZ = std::max(1, int(std::floor(size.z / spacing)) + 1);
  int nPoints = pointsX * pointsY * pointsZ;
  // ---------------------------

  std::vector<glm::vec3> samples;
  std::vector<std::optional<glm::vec3>> mask(nPoints, std::nullopt);

  volume_apply(
      [&](auto idx, const auto &p) {
        if (filter(p)) {
          return;
        }
        mask[idx] = p;
      },
      bounds, spacing);

  for (const auto active : mask) {
    if (!active) {
      continue;
    }
    samples.push_back(*active);
  }

  return samples;
}

}; // namespace detail

using SDF = structuring_elements::PointStructuringElement::SDF;

template <structuring_elements::Operation op>
auto morph(const APSS &original_apss, const ::detail::Bounds &bounds,
           const SDF &sdf, float sigmaP, float pse_scale) {

  const auto denseSampling = morphology::detail::sampleBoxVolume(
      bounds, sigmaP, [&](const auto &sample) {
        auto dist = original_apss.evaluate_surface(sample);

        if constexpr (op == structuring_elements::Erosion) {
          dist += pse_scale; // Sample close to dilation surface
        } else if constexpr (op == structuring_elements::Dilation) {
          dist -= pse_scale;
        }

        return std::abs(dist) > 2 * sigmaP;
      });

  drawPointCloud("dense sampling", denseSampling)->setEnabled(false);
  std::cout << "Dense sampling count = " << denseSampling.size() << std::endl;

  auto morphed_points = denseSampling;
  auto morphed_normals = std::vector<PointCloud::Normal>(morphed_points.size());
  
  // now we want to project the extruded points down onto the dilation

  unsigned numThreads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  std::atomic<size_t> pointsDone;

  auto worker = [&](size_t start, size_t end) {
    auto projected = 0;
    static const auto batch = 1000;
    for (auto i = start; i < end; i++) {
      // now we iteratively project
      auto [p, n] = structuring_elements::project_iterative<op>(
          original_apss, morphed_points[i], sdf, pse_scale);

      morphed_points[i] = p;
      morphed_normals[i] = n;

      projected += 1;

      if (projected == batch || i == end - 1) {
        pointsDone.fetch_add(projected, std::memory_order_relaxed);
        projected = 0;
      }
    }
  };

  for (auto threadIdx = 0; threadIdx < numThreads; threadIdx++) {
    auto chunkSize = std::ceil(morphed_points.size() / numThreads);

    const auto start = chunkSize * threadIdx;
    auto end = start + chunkSize;

    // last thread works until end
    if (threadIdx == numThreads - 1) {
      end = morphed_points.size();
    }

    threads.emplace_back(worker, start, end);
  }

  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "\rMorphological Projection: "
              << (100.0 * pointsDone / morphed_points.size()) << "%   "
              << std::flush;
    if (pointsDone >= morphed_points.size()) {
      std::cout << std::endl;
      break;
    }
  }

  for (auto &t : threads) {
    t.join();
  }

  return PointCloud(morphed_points, morphed_normals);
};

} // namespace morphology
