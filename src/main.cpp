#include "APSS.hpp"
#include "PointCloud.hpp"
#include "Subsample.hpp"
#include <nanoflann.hpp>
#include <polyscope/implicit_helpers.h>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/types.h>
#include <polyscope/utilities.h>
#include <polyscope/volume_grid.h>

struct PointStructuringElement {

  using Position = glm::vec3;
  using SDF = std::function<float(const Position &)>;

  float s;    // scale
  Position c; // center
  SDF B;      // sdf of its shape

  PointStructuringElement(float _s, Position _c, SDF _B)
      : s(_s), c(_c), B(_B) {};

  float operator()(const Position &x) const noexcept {
    return s * B((x - c) / s);
  }
};

auto sphereSDF = [](const glm::vec3 &p) { return glm::length(p) - 1.0f; };

auto torusSDF = [](const glm::vec3 &p) {
  // TODO: Make a generator for different values of t
  static const auto t = glm::vec2(1.0, 0.3);
  glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)) - t.x, p.y);
  return glm::length(q) - t.y;
};

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

int main() {

  // Options
  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();

  const auto hand = PointCloud("./resources/hand.ply");
  const auto handSubsampled = poissonDiskSubsample(hand, .01f);
  std::cout << "Reduction = "
            << handSubsampled.positions.size() /
                   static_cast<float>(hand.positions.size())
            << std::endl;

  const auto apss = APSS(handSubsampled);
  const auto max_spacing = handSubsampled.maximum_point_spacing();
  const auto avg_spacing = handSubsampled.average_point_spacing();
  const auto min_spacing = handSubsampled.minimum_point_spacing();

  std::cout << "max spacing: " << max_spacing << std::endl;
  std::cout << "avg spacing: " << avg_spacing << std::endl;
  std::cout << "min spacing: " << min_spacing << std::endl;

  auto bounds = detail::computeBoundingBox(handSubsampled.positions, 1.2);

  const auto apssSDF = [&apss, max_spacing](const glm::vec3 &p) {
    return apss.evaluate_surface(p, .5f);
  };

  addVolumeGrid(bounds, apssSDF, 100);

  std::cout << "Done" << std::endl;

  polyscope::show();
}
