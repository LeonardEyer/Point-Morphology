
#include "APSS.hpp"
#include "PointCloud.hpp"
#include "Resample.hpp"
#include "Subsample.hpp"

#include <nanoflann.hpp>
#include <polyscope/implicit_helpers.h>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/types.h>
#include <polyscope/utilities.h>
#include <polyscope/volume_grid.h>

template <typename SDFFunc>
void addVolumeGrid(const detail::Bounds &bounds, const SDFFunc &sdf,
                   const size_t resolution = 50) {

  uint32_t dimX = resolution;
  uint32_t dimY = resolution;
  uint32_t dimZ = resolution;

  std::vector<float> values(dimX * dimY * dimZ);

  const auto xSpacing = (bounds.second.x - bounds.first.x) / (dimX - 1);
  const auto ySpacing = (bounds.second.y - bounds.first.y) / (dimY - 1);
  const auto zSpacing = (bounds.second.z - bounds.first.z) / (dimZ - 1);

  for (int ix = 0; ix < dimX; ++ix) {
    for (int iy = 0; iy < dimY; ++iy) {
      for (int iz = 0; iz < dimZ; ++iz) {
        auto p = glm::vec3(bounds.first.x + ix * xSpacing,
                           bounds.first.y + iy * ySpacing,
                           bounds.first.z + iz * zSpacing);

        const size_t idx = static_cast<size_t>(iz) * dimY * dimX +
                           static_cast<size_t>(iy) * dimX +
                           static_cast<size_t>(ix);

        values[idx] = sdf(p);
      }
    }
  }

  polyscope::VolumeGrid *psGrid = polyscope::registerVolumeGrid(
      "test grid", {dimX, dimY, dimZ}, bounds.first, bounds.second);
  psGrid->setEdgeWidth(1.0);

  polyscope::VolumeGridNodeScalarQuantity *qNode =
      psGrid->addNodeScalarQuantity("sdf node", values);
  qNode->setEnabled(true);

  qNode->setGridcubeVizEnabled(false);
  qNode->setIsosurfaceLevel(0.0);
  qNode->setIsosurfaceVizEnabled(true);
}

void drawPointCloud(const PointCloud &p) {
  auto *handCloud = polyscope::registerPointCloud("positions", p.positions);
  handCloud->addVectorQuantity("normals", p.normals);
  handCloud->setPointRadius(0.0002);
  handCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);
}

void log_point_cloud_stats(const PointCloud &cloud) {

  const auto max_spacing = cloud.maximum_point_spacing();
  const auto avg_spacing = cloud.average_point_spacing();
  const auto min_spacing = cloud.minimum_point_spacing();

  std::cout << "max spacing: " << max_spacing << std::endl;
  std::cout << "avg spacing: " << avg_spacing << std::endl;
  std::cout << "min spacing: " << min_spacing << std::endl;
}

int main() {

  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;
  polyscope::init();

  const auto hand = PointCloud("./resources/hand.ply");
  auto handSubsampled = poissonDiskSubsample(hand, .1f);
  std::cout << "Reduction = "
            << 1.0f - handSubsampled.positions.size() /
                          static_cast<float>(hand.positions.size())
            << std::endl;

  log_point_cloud_stats(handSubsampled);
  resample(handSubsampled, .8f, 1u);

  const auto apss = APSS(handSubsampled);

  auto bounds = detail::computeBoundingBox(handSubsampled.positions, 1.2);

  const auto apssSDF = [&apss](const glm::vec3 &p) {
    const auto fitted = apss.fit(p, 1.0f);

    return std::visit([&p](const auto &fit) { return distance(fit, p); },
                      fitted);
  };

  addVolumeGrid(bounds, apssSDF, 50);

  std::cout << "Done" << std::endl;
  polyscope::show();
}
