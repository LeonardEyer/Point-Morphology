#include <glm/geometric.hpp>
#include <polyscope/implicit_helpers.h>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/types.h>
#include <polyscope/utilities.h>
#include <polyscope/volume_grid.h>

auto torusSDF = [](glm::vec3 p) {
  // TODO: Make a generator for different values of t
  static const auto t = glm::vec2(1.0, 0.3);
  glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)) - t.x, p.y);
  return glm::length(q) - t.y;
};

void addVolumeGrid() {

  uint32_t dimX = 50;
  uint32_t dimY = 50;
  uint32_t dimZ = 50;

  glm::vec3 bound_low{-3., -3., -3.};
  glm::vec3 bound_high{3., 3., 3.};

  polyscope::VolumeGrid *psGrid = polyscope::registerVolumeGrid(
      "test grid", {dimX, dimY, dimZ}, bound_low, bound_high);

  psGrid->setEdgeWidth(1.0);

  polyscope::VolumeGridNodeScalarQuantity *qNode =
      psGrid->addNodeScalarQuantityFromCallable("torus sdf node", torusSDF);
  qNode->setEnabled(true);

  qNode->setGridcubeVizEnabled(false);
  qNode->setIsosurfaceLevel(0.0);
  qNode->setIsosurfaceVizEnabled(true);
}

void callback() {
  if (ImGui::Button("add implicits")) {
    polyscope::ImplicitRenderOpts opts;
    polyscope::ImplicitRenderMode mode =
        polyscope::ImplicitRenderMode::SphereMarch;
    opts.subsampleFactor = 2; // downsample the rendering

    // render the implicit isosurface from the current viewport
    polyscope::DepthRenderImageQuantity *img =
        polyscope::renderImplicitSurface("torus sdf", torusSDF, mode, opts);
  }
}

int main() {

  // Options
  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();

  addVolumeGrid();

  polyscope::state::userCallback = callback;

  polyscope::show();
}
