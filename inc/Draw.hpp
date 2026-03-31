#pragma once

#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/volume_grid.h>
#include <thread>

#include "PointCloud.hpp"

namespace draw {

static const std::string camera_json =
    R"(
{"farClipRatio":20.0,"fov":35.0,"nearClipRatio":0.005,"projectionMode":"Perspective","viewMat":[0.815649330615997,4.4703483581543e-08,0.578546643257141,0.0,0.165076032280922,0.958429634571075,-0.232728332281113,0.0,-0.554496288299561,0.285328894853592,0.781742453575134,-103.623413085938,0.0,0.0,0.0,1.0],"windowHeight":1018,"windowWidth":1024}
)";

inline auto drawPointCloud(
    std::string name, const PointCloud &p,
    polyscope::PointRenderMode mode = polyscope::PointRenderMode::Quad) {
  auto cloud = polyscope::registerPointCloud(name, p.positions);
  cloud->addVectorQuantity("normals", p.normals);
  cloud->setPointRadius(0.002);
  cloud->setPointRenderMode(mode);

  return cloud;
}

inline auto drawPointCloud(
    std::string name, const std::vector<PointCloud::Position> &p,
    const std::vector<PointCloud::Normal> &n,
    polyscope::PointRenderMode mode = polyscope::PointRenderMode::Quad) {
  auto cloud = polyscope::registerPointCloud(name, p);
  cloud->addVectorQuantity("normals", n);
  cloud->setPointRadius(0.002);
  cloud->setPointRenderMode(mode);

  return cloud;
}

inline auto drawPointCloud(
    std::string name, const std::vector<PointCloud::Position> &p,
    polyscope::PointRenderMode mode = polyscope::PointRenderMode::Quad) {
  auto *cloud = polyscope::registerPointCloud(name, p);
  cloud->setPointRadius(0.002);
  cloud->setPointRenderMode(mode);

  return cloud;
}

namespace detail {

template <typename SDFFunc>
void computeVolumeGrid(const glm::vec3 &minBound, const glm::vec3 &maxBound,
                       uint32_t dimX, uint32_t dimY, uint32_t dimZ,
                       float spacingX, float spacingY, float spacingZ,
                       const SDFFunc &sdf, std::vector<float> &values,
                       std::atomic<size_t> &voxelsDone) {

  unsigned numThreads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;

  auto worker = [&](size_t zStart, size_t zEnd) {
    for (size_t iz = zStart; iz < zEnd; ++iz) {
      for (size_t iy = 0; iy < dimY; ++iy) {
        for (size_t ix = 0; ix < dimX; ++ix) {
          size_t idx = iz * dimY * dimX + iy * dimX + ix;

          glm::vec3 p(minBound.x + ix * spacingX, minBound.y + iy * spacingY,
                      minBound.z + iz * spacingZ);

          p.x = std::min(p.x, maxBound.x);
          p.y = std::min(p.y, maxBound.y);
          p.z = std::min(p.z, maxBound.z);

          values[idx] = sdf(p);
          voxelsDone.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  };

  size_t chunk = dimZ / numThreads;
  size_t zStart = 0;
  for (unsigned t = 0; t < numThreads; ++t) {
    size_t zEnd = (t == numThreads - 1) ? dimZ : zStart + chunk;
    threads.emplace_back(worker, zStart, zEnd);
    zStart = zEnd;
  }

  size_t totalVoxels = dimX * dimY * dimZ;
  while (voxelsDone < totalVoxels) {
    std::cout << "\rComputing Grid Volume Voxels (Progress): "
              << (100.0 * voxelsDone / totalVoxels) << "%   " << std::flush;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  for (auto &t : threads)
    t.join();
  std::cout << "\rComputing Grid Volume Voxels (Progress): 100%   \n";
  voxelsDone = 0;
}

} // namespace detail

template <typename SDFFunc>
void addVolumeGridByResolution(const std::string &name,
                               const ::detail::Bounds &bounds,
                               const SDFFunc &sdf,
                               const size_t resolution = 50) {

  auto &[minBound, maxBound] = bounds;

  uint32_t dimX = resolution;
  uint32_t dimY = resolution;
  uint32_t dimZ = resolution;

  float spacingX = (maxBound.x - minBound.x) / (dimX - 1);
  float spacingY = (maxBound.y - minBound.y) / (dimY - 1);
  float spacingZ = (maxBound.z - minBound.z) / (dimZ - 1);

  std::vector<float> values(dimX * dimY * dimZ);
  static std::atomic<size_t> voxelsDone{0};

  detail::computeVolumeGrid(minBound, maxBound, dimX, dimY, dimZ, spacingX,
                            spacingY, spacingZ, sdf, values, voxelsDone);

  auto *psGrid = polyscope::registerVolumeGrid(name, {dimX, dimY, dimZ},
                                               minBound, maxBound);
  psGrid->setEdgeWidth(0);
  auto *qNode = psGrid->addNodeScalarQuantity("sdf node", values);
  qNode->setEnabled(true);
  qNode->setIsosurfaceLevel(0.0);
  qNode->setIsosurfaceVizEnabled(true);
  qNode->setGridcubeVizEnabled(false);
}

} // namespace draw
