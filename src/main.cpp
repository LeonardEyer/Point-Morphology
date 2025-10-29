#include "APSS.hpp"
#include "PointCloud.hpp"
#include "glm/ext/vector_float3.hpp"
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

template <typename SDFFunc> void addVolumeGrid(const SDFFunc &sdf) {

  uint32_t dimX = 100;
  uint32_t dimY = 100;
  uint32_t dimZ = 100;

  glm::vec3 bound_low{-3., -3., -3.};
  glm::vec3 bound_high{3., 3., 3.};

  polyscope::VolumeGrid *psGrid = polyscope::registerVolumeGrid(
      "test grid", {dimX, dimY, dimZ}, bound_low, bound_high);

  psGrid->setEdgeWidth(1.0);

  polyscope::VolumeGridNodeScalarQuantity *qNode =
      psGrid->addNodeScalarQuantityFromCallable("sdf node", sdf);
  qNode->setEnabled(true);

  qNode->setGridcubeVizEnabled(false);
  qNode->setIsosurfaceLevel(0.0);
  qNode->setIsosurfaceVizEnabled(true);
}

int main() {

  // Options
  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();

  const auto hand = PointCloud("./resources/hand.ply");

  const auto apss = APSS(hand);

  auto *handCloud =
      polyscope::registerPointCloud("hand positions", hand.positions);

  handCloud->addVectorQuantity("normals", hand.normals);
  handCloud->setPointRadius(0.0002);
  handCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);

  const auto apssSDF = [&apss](const glm::vec3 &p) {
    return apss.evaluate_surface(p + glm::vec3(2.5, 2.5, 2.5), 2.f);
  };

  addVolumeGrid(apssSDF);

  polyscope::show();
}
